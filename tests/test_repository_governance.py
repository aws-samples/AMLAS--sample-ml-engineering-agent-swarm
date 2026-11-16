from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import runpy
import string
import subprocess
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_repository_governance.py"
PUBLIC_ORACLE_METADATA_FIXTURE_SHA256 = (
    "f0a28bb14aff546956ff027ba9b9bbc7417c022eb315eb9c4fa3c668d8a08e1c"  # pragma: allowlist secret
)
PUBLIC_ORACLE_DIGEST_FIXTURE = (
    "016242643122db3e21319d9a28d85c518e4dee3ec44846c783df43a8ba516206"  # pragma: allowlist secret
)
PUBLIC_REVISION_FIXTURE = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"  # pragma: allowlist secret

SPEC = importlib.util.spec_from_file_location("verify_repository_governance", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
governance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(governance)


ORACLE_ANNOTATION_PATHS = (
    "scripts/verify_repository_governance.py",
    "src/amlas/evaluation/coding_oracle_catalog.py",
    "tests/test_paired_coding_publication.py",
    "tests/test_repository_governance.py",
)


def public_oracle_annotation_errors(contents: dict[str, str]) -> list[str]:
    """Freeze the exact reviewed annotation lines, without importing oracle code."""
    catalog_path = ORACLE_ANNOTATION_PATHS[1]
    metadata = next(
        ast.literal_eval(node.value)
        for node in ast.parse(contents[catalog_path]).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_EXPECTED_JSON" for target in node.targets)
    )
    if hashlib.sha256(metadata.encode()).hexdigest() != PUBLIC_ORACLE_METADATA_FIXTURE_SHA256:
        return ["public oracle metadata bytes changed"]
    rows = json.loads(metadata)
    pins = {
        value
        for row in rows
        for key, value in row.items()
        if key.endswith("_sha256") and re.search(r"(?<!\d)\d{12}(?!\d)", value)
    }
    assert len(rows) == 164 and len(pins) == 21
    marker = "  # pragma: allowlist secret"
    original_sha = (
        "1d49078ba3e2b196b9344535bef34a43021f038fad9561d6ee7c53450609a6a2"  # pragma: allowlist secret
    )
    public_revision_name = "_".join(("PUBLIC", "REVISION", "FIXTURE"))
    fixture_text = (REPO_ROOT / "tests/test_qwen_sft_reload_resources.py").read_text()
    fixture_checksum = next(
        ast.literal_eval(node.value)
        for node in ast.parse(fixture_text).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "TRAIN_REQUEST" for target in node.targets)
    )
    expected = {
        ORACLE_ANNOTATION_PATHS[0]: {
            f'"{digest}",' + marker for digest in pins | {PUBLIC_REVISION_FIXTURE, fixture_checksum}
        },
        catalog_path: {
            "_EXPECTED_JSON = " + repr(metadata) + marker,
            f'"{original_sha}"' + marker,
        },
        ORACLE_ANNOTATION_PATHS[2]: {f'"{original_sha}"' + marker},
        ORACLE_ANNOTATION_PATHS[3]: {
            f'"{PUBLIC_ORACLE_METADATA_FIXTURE_SHA256}"' + marker,
            f'"{PUBLIC_ORACLE_DIGEST_FIXTURE}"' + marker,
            f'{public_revision_name} = "{PUBLIC_REVISION_FIXTURE}"' + marker,
            'token = "glpat" + "-' + string.ascii_lowercase + '"' + marker,
            f'"{original_sha}"' + marker,
        },
    }
    return [
        path
        for path in ORACLE_ANNOTATION_PATHS
        if {line.strip() for line in contents[path].splitlines() if line.rstrip().endswith(marker)}
        != expected[path]
    ]


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def locked_version(package_name: str) -> tuple[int, ...]:
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    for package in lock["package"]:
        if package["name"] == package_name:
            return tuple(int(part) for part in package["version"].split("."))
    raise AssertionError(f"{package_name} is missing from uv.lock")


class ContentScanTests(unittest.TestCase):
    def test_generic_vcs_names_are_public_safe(self) -> None:
        text = "GitHub main is the authoritative public release tree. Development happens on GitLab branches."
        self.assertEqual([], governance.scan_text("AGENTS.md", text))

    def test_internal_coordinates_are_rejected(self) -> None:
        internal_host = ".".join(("review", "corp", "example"))
        findings = governance.scan_text("config.txt", f"remote={internal_host}")
        self.assertIn("internal-host", {finding.code for finding in findings})

    def test_secret_prefix_is_rejected(self) -> None:
        token = "glpat" + "-abcdefghijklmnopqrstuvwxyz"  # pragma: allowlist secret
        findings = governance.scan_text("example.env", f"TOKEN={token}")
        self.assertIn("secret-token", {finding.code for finding in findings})

    def test_unknown_account_id_is_rejected(self) -> None:
        account_id = "123456" + "789012"
        findings = governance.scan_text("config.py", f'ACCOUNT_ID = "{account_id}"')
        self.assertIn("unknown-account-id", {finding.code for finding in findings})

    def test_exact_reviewed_public_model_revision_is_not_an_account(self) -> None:
        revision = PUBLIC_REVISION_FIXTURE
        for text in (
            f'REVISION = "{revision}"',
            f"https://huggingface.co/Qwen/Qwen3.5-9B/resolve/{revision}/config.json",
            f"Public model commit `{revision}`.",
        ):
            with self.subTest(text=text):
                self.assertEqual([], governance.scan_text("model_manifest.py", text))

    def test_public_revision_exception_does_not_allow_numeric_fragment(self) -> None:
        fragment = "202236" + "235762"
        findings = governance.scan_text("config.py", f'ACCOUNT_ID = "{fragment}"')
        self.assertIn("unknown-account-id", {finding.code for finding in findings})

    def test_public_revision_exception_requires_exact_hex_token(self) -> None:
        revision = PUBLIC_REVISION_FIXTURE
        for value in ("a" + revision, revision + "f", revision[:-1], revision.upper()):
            with self.subTest(value=value):
                findings = governance.scan_text("config.py", f'REVISION = "{value}"')
                self.assertIn("unknown-account-id", {finding.code for finding in findings})

    def test_public_revision_does_not_hide_other_private_values(self) -> None:
        revision = PUBLIC_REVISION_FIXTURE
        account = "123456" + "789012"
        findings = governance.scan_text("config.py", f'REVISION="{revision}" ACCOUNT="{account}"')
        self.assertIn("unknown-account-id", {finding.code for finding in findings})
        findings = governance.scan_text("config.py", revision, (revision,))
        self.assertIn("private-literal", {finding.code for finding in findings})

    def test_fixed_public_oracle_catalogue_hashes_are_not_accounts(self) -> None:
        path = "src/amlas/evaluation/coding_oracle_catalog.py"
        self.assertEqual([], governance.scan_text(path, (REPO_ROOT / path).read_text()))

    def test_exact_reviewed_public_oracle_digest_is_not_an_account(self) -> None:
        digest = PUBLIC_ORACLE_DIGEST_FIXTURE
        for text in (f'HASH = "{digest}"', f"Public oracle digest `{digest}`."):
            with self.subTest(text=text):
                self.assertEqual([], governance.scan_text("metadata.py", text))

    def test_public_oracle_exception_requires_exact_hex_token(self) -> None:
        digest = PUBLIC_ORACLE_DIGEST_FIXTURE
        for value in ("a" + digest, digest + "f", digest[:-1], digest.upper(), digest[:-1] + "7"):
            with self.subTest(value=value):
                findings = governance.scan_text("metadata.py", f'HASH = "{value}"')
                self.assertIn("unknown-account-id", {item.code for item in findings})

    def test_public_oracle_exception_does_not_allow_numeric_fragments(self) -> None:
        fragment = "016242" + "643122"
        for value in (fragment, "a" * 26 + fragment + "b" * 26):
            with self.subTest(value=value):
                findings = governance.scan_text("metadata.py", f'VALUE = "{value}"')
                self.assertIn("unknown-account-id", {item.code for item in findings})

    def test_public_oracle_digest_does_not_hide_private_values(self) -> None:
        digest = PUBLIC_ORACLE_DIGEST_FIXTURE
        account = "123456" + "789012"
        for path in ("metadata.py", "src/amlas/evaluation/coding_oracle_catalog.py"):
            with self.subTest(path=path):
                findings = governance.scan_text(path, f'HASH="{digest}" ACCOUNT="{account}"')
                self.assertIn("unknown-account-id", {item.code for item in findings})
                findings = governance.scan_text(path, digest, (digest,))
                self.assertIn("private-literal", {item.code for item in findings})

    def test_public_oracle_allowlist_annotations_have_exact_reviewed_bytes(self) -> None:
        contents = {path: (REPO_ROOT / path).read_text() for path in ORACLE_ANNOTATION_PATHS}
        self.assertEqual([], public_oracle_annotation_errors(contents))

    def test_changed_or_added_annotated_public_hash_is_not_reviewed(self) -> None:
        contents = {path: (REPO_ROOT / path).read_text() for path in ORACLE_ANNOTATION_PATHS}
        catalog_path = ORACLE_ANNOTATION_PATHS[1]
        changed = dict(contents)
        changed[catalog_path] = contents[catalog_path].replace(
            PUBLIC_ORACLE_DIGEST_FIXTURE, PUBLIC_ORACLE_DIGEST_FIXTURE[:-1] + "7", 1
        )
        self.assertEqual(["public oracle metadata bytes changed"], public_oracle_annotation_errors(changed))
        extra_digest = hashlib.sha256(b"neutral unreviewed metadata fixture").hexdigest()
        changed = dict(contents)
        changed[catalog_path] += f'\nEXTRA = "{extra_digest}"  # pragma: allowlist secret\n'
        self.assertIn(catalog_path, public_oracle_annotation_errors(changed))

    def test_annotation_does_not_hide_private_account_or_strict_entropy(self) -> None:
        from detect_secrets.plugins.high_entropy_strings import HexHighEntropyString

        account = "123456" + "789012"
        line = f'ACCOUNT = "{account}"  # pragma: allowlist secret'
        self.assertIn("unknown-account-id", {item.code for item in governance.scan_text("metadata.py", line)})
        digest = hashlib.sha256(b"neutral unreviewed metadata fixture").hexdigest()
        line = f'VALUE = "{digest}"  # pragma: allowlist secret'
        self.assertTrue(HexHighEntropyString().analyze_line("metadata.py", line))

    def test_approved_aws_publisher_id_is_allowed(self) -> None:
        publisher_id = "246618" + "743249"
        self.assertEqual(
            [],
            governance.scan_text(
                "src/amlas/config/sagemaker_images.py",
                f'DLC_ACCOUNT = "{publisher_id}"',
            ),
        )

    def test_regional_image_catalog_uses_only_reviewed_publishers(self) -> None:
        # Regional image selection must not bypass the publication account allowlist.
        catalog = runpy.run_path(str(REPO_ROOT / "src/amlas/config/_sagemaker_registry.py"))
        names = (
            "SKLEARN_DLC_ACCOUNTS",
            "XGBOOST_DLC_ACCOUNTS",
            "CLARIFY_DLC_ACCOUNTS",
            "MODEL_MONITOR_DLC_ACCOUNTS",
            "PYTORCH_DLC_ACCOUNTS",
        )
        for name in names:
            for region, publisher in catalog[name].items():
                with self.subTest(catalog=name, region=region):
                    self.assertEqual(
                        [],
                        governance.scan_text(
                            "src/amlas/config/_sagemaker_registry.py", f'PUBLISHER = "{publisher}"'
                        ),
                    )

    def test_generated_artifact_path_is_rejected(self) -> None:
        findings = governance.scan_path("output/model.pkl")
        self.assertIn("generated-artifact", {finding.code for finding in findings})


class GitReferenceTests(unittest.TestCase):
    def test_unrelated_commits_with_identical_trees_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            git(repo.parent, "init", str(repo))
            git(repo, "config", "user.name", "Governance Test")
            git(repo, "config", "user.email", "governance@example.com")

            (repo / "payload.txt").write_text("same tree\n", encoding="utf-8")
            git(repo, "add", "payload.txt")
            git(repo, "commit", "-m", "internal history")
            internal_ref = git(repo, "rev-parse", "HEAD")

            git(repo, "switch", "--orphan", "public-root")
            (repo / "payload.txt").write_text("same tree\n", encoding="utf-8")
            git(repo, "add", "payload.txt")
            git(repo, "commit", "-m", "public root")
            public_ref = git(repo, "rev-parse", "HEAD")

            self.assertNotEqual(internal_ref, public_ref)
            self.assertTrue(governance.refs_have_equal_trees(repo, internal_ref, public_ref))

            (repo / "payload.txt").write_text("different tree\n", encoding="utf-8")
            git(repo, "commit", "-am", "change public tree")
            changed_ref = git(repo, "rev-parse", "HEAD")
            self.assertFalse(governance.refs_have_equal_trees(repo, internal_ref, changed_ref))


class DependencyFloorTests(unittest.TestCase):
    def test_security_floors_are_locked(self) -> None:
        self.assertGreaterEqual(locked_version("anyio"), (4, 14, 2))
        self.assertGreaterEqual(locked_version("gitpython"), (3, 1, 60))


class PackagingPolicyTests(unittest.TestCase):
    def test_sdist_excludes_repository_governance_files(self) -> None:
        project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        excluded = set(project["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"])
        self.assertTrue(
            {
                "/.github",
                "/.gitlab-ci.yml",
                "/.secrets.baseline",
                "/AGENTS.md",
                "/tests",
                "/scripts/verify_repository_governance.py",
            }.issubset(excluded)
        )


class RepositoryScanTests(unittest.TestCase):
    def test_release_governance_files_exist(self) -> None:
        required = {
            "AGENTS.md",
            ".github/workflows/public-release.yml",
            ".gitlab-ci.yml",
            "scripts/verify_repository_governance.py",
            "tests/test_repository_governance.py",
        }
        self.assertEqual([], governance.missing_required_paths(REPO_ROOT, required))

    def test_candidate_tree_is_public_safe(self) -> None:
        self.assertEqual([], governance.scan_repository(REPO_ROOT))


class ImmutablePublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        git(self.root, "init", str(self.repo))
        git(self.repo, "config", "user.name", "Governance Test")
        git(self.repo, "config", "user.email", "governance@example.com")

    def stage(self, path: str, content: str) -> None:
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        git(self.repo, "add", path)

    def test_lockfile_and_baseline_cannot_hide_credentials(self) -> None:
        # Regression: the full-tree gate previously skipped these two tracked files.
        for name in ("uv.lock", ".secrets.baseline"):
            self.stage(name, "token=" + "glpat" + "-" + "x" * 32)
        found = {(item.path, item.code) for item in governance.scan_repository(self.repo)}
        self.assertTrue({("uv.lock", "secret-token"), (".secrets.baseline", "secret-token")} <= found)

    def test_clean_working_copy_cannot_mask_staged_secret(self) -> None:
        # Regression: publication must inspect Git's candidate bytes, not edited disk files.
        self.stage("payload.txt", "glpat" + "-" + "x" * 32)
        (self.repo / "payload.txt").write_text("clean working copy", encoding="utf-8")
        found = {(item.path, item.code) for item in governance.scan_repository(self.repo)}
        self.assertIn(("payload.txt", "secret-token"), found)

    def test_symlink_is_rejected_without_following_target(self) -> None:
        # Regression: links previously followed local files or were silently skipped.
        target = self.root / "outside.txt"
        target.write_text("glpat" + "-" + "x" * 32, encoding="utf-8")
        (self.repo / "linked.txt").symlink_to(target)
        git(self.repo, "add", "linked.txt")
        codes = {item.code for item in governance.scan_repository(self.repo) if item.path == "linked.txt"}
        self.assertIn("unsupported-git-mode", codes)
        self.assertNotIn("secret-token", codes)

    def test_private_denylist_stays_outside_public_source(self) -> None:
        # Regression: actual private identifiers were embedded in the public checker.
        self.stage("payload.txt", "private-example-alias")
        denylist = self.root / "release-denylist.txt"
        denylist.write_text("private-example-alias\n", encoding="utf-8")
        with patch.dict(os.environ, {"AMLAS_RELEASE_DENY_LITERALS_FILE": str(denylist)}):
            findings = governance.scan_repository(self.repo)
        self.assertIn(("payload.txt", "private-literal"), {(item.path, item.code) for item in findings})
        self.assertNotIn("private-example-alias", str(findings))

    def test_public_dependency_update_automation_requires_removal(self) -> None:
        # Regression: automatic public PR branches bypass the exact-tree approval gate.
        self.stage(".github/dependabot.yml", "version: 2\nupdates: []\n")
        findings = governance.scan_repository(self.repo)
        self.assertIn("public-write-automation", {item.code for item in findings})


class PublicationReviewRegressionTests(unittest.TestCase):
    setUp = ImmutablePublicationTests.setUp
    stage = ImmutablePublicationTests.stage

    def test_private_filenames_are_rejected_without_echoing_them(self) -> None:
        # Regression: the private denylist previously covered only blob contents.
        alias = "private-example-alias"
        self.stage(f"notes/{alias}.md", "public text")
        git(self.repo, "commit", "-m", "candidate")
        denylist = self.root / "release-denylist.txt"
        denylist.write_text(alias + "\n", encoding="utf-8")
        with patch.dict(os.environ, {"AMLAS_RELEASE_DENY_LITERALS_FILE": str(denylist)}):
            for ref in (None, "HEAD"):
                with self.subTest(ref=ref):
                    findings = governance.scan_repository(self.repo, ref)
                    self.assertIn("private-literal", {item.code for item in findings})
                    self.assertNotIn(alias, str(findings))

    def test_customer_account_in_lockfile_is_rejected(self) -> None:
        # Regression: lockfiles had a blanket account-ID exemption.
        account = "123456" + "789012"
        self.stage("uv.lock", f"# customer account: {account}\n")
        findings = governance.scan_repository(self.repo)
        self.assertIn(("uv.lock", "unknown-account-id"), {(item.path, item.code) for item in findings})

    def test_integrity_exemptions_cover_only_valid_hash_contexts(self) -> None:
        account = "123456" + "789012"
        digest = "a" * 26 + account + "b" * 26
        package_hash = "a" * 24 + account + "b" * 24
        text = f'hash = "sha256:{digest}"\nurl = "https://files.pythonhosted.org/packages/ab/cd/{package_hash}/file.whl"'
        self.assertEqual([], governance.scan_text("uv.lock", text))
        findings = governance.scan_text("uv.lock", text + f"\n# account: {account}")
        self.assertIn("unknown-account-id", {item.code for item in findings})

    def test_placeholder_substrings_cannot_hide_tokens(self) -> None:
        # Regression: any secret-shaped string containing these words was exempted.
        tokens = (
            "ghp" + "_" + "a" * 8 + "placeholder" + "b" * 12,
            "sk" + "-" + "a" * 8 + "your-real-key" + "b" * 12,
        )
        for token in tokens:
            with self.subTest(kind=token[:3]):
                self.assertIn(
                    "secret-token", {item.code for item in governance.scan_text("config.txt", token)}
                )

    def test_private_account_diagnostics_do_not_echo_the_value(self) -> None:
        account = "123456" + "789012"
        findings = governance.scan_text("config.txt", account, (account,))
        self.assertIn("unknown-account-id", {item.code for item in findings})
        self.assertNotIn(account, str(findings))


class DoxAliasPublicationTests(unittest.TestCase):
    setUp = ImmutablePublicationTests.setUp
    stage = ImmutablePublicationTests.stage

    def test_exact_root_claude_alias_is_public_safe(self) -> None:
        self.stage("AGENTS.md", "# Repository contract\n")
        (self.repo / "CLAUDE.md").symlink_to("AGENTS.md")
        git(self.repo, "add", "CLAUDE.md")
        findings = governance.scan_repository(self.repo)
        self.assertEqual([], [x for x in findings if x.path == "CLAUDE.md"])

    def test_alias_uses_index_bytes_and_requires_regular_tracked_target(self) -> None:
        self.stage("AGENTS.md", "# Repository contract\n")
        (self.repo / "CLAUDE.md").symlink_to("AGENTS.md")
        git(self.repo, "add", "CLAUDE.md")
        (self.repo / "CLAUDE.md").unlink()
        (self.repo / "CLAUDE.md").symlink_to("outside.md")
        self.assertEqual([], [x for x in governance.scan_repository(self.repo) if x.path == "CLAUDE.md"])
        git(self.repo, "rm", "--cached", "AGENTS.md")
        self.assertIn(
            "unsupported-git-mode",
            {x.code for x in governance.scan_repository(self.repo) if x.path == "CLAUDE.md"},
        )

    def test_wrong_nested_and_nonregular_aliases_are_rejected(self) -> None:
        for path, target in [
            ("CLAUDE.md", "../AGENTS.md"),
            ("CLAUDE.md", "AGENTS.md\n"),
            ("nested/CLAUDE.md", "../AGENTS.md"),
        ]:
            with self.subTest(path=path, target=target):
                self.stage("AGENTS.md", "# Repository contract\n")
                link = self.repo / path
                link.parent.mkdir(parents=True, exist_ok=True)
                if link.is_symlink():
                    link.unlink()
                link.symlink_to(target)
                git(self.repo, "add", path)
                self.assertIn(
                    "unsupported-git-mode",
                    {x.code for x in governance.scan_repository(self.repo) if x.path == path},
                )

    def test_symlink_target_cannot_replace_regular_root_agents(self) -> None:
        self.stage("target.md", "# Public contract\n")
        (self.repo / "AGENTS.md").symlink_to("target.md")
        (self.repo / "CLAUDE.md").symlink_to("AGENTS.md")
        git(self.repo, "add", "AGENTS.md", "CLAUDE.md")
        self.assertIn(
            "unsupported-git-mode",
            {x.code for x in governance.scan_repository(self.repo) if x.path == "CLAUDE.md"},
        )


if __name__ == "__main__":
    unittest.main()
