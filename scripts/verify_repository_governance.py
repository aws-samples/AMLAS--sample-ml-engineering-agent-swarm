#!/usr/bin/env python3
"""Verify public-release safety and GitHub/GitLab main-tree equivalence."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple


class Finding(NamedTuple):
    path: str
    code: str
    detail: str


PUBLIC_AWS_ACCOUNT_IDS = {
    "000000000000",  # Deliberately invalid placeholder.
    "111122223333",  # AWS documentation example.
    "081040173940",
    "141502667606",
    "159807026194",
    "246618743249",
    "257758044811",
    "266458841044",
    "306415355426",
    "432418664414",
    "477331159723",
    "568282634449",
    "683313688378",
    "763104351884",
    # Regional AWS SageMaker publisher accounts used by the five image catalogs.
    # Verified against aws/sagemaker-python-sdk commit
    # 0495265923351eb4c0a3ceaa635acabe2892b050 and its image_uri_config JSON files.
    "001633400207",
    "007051062584",
    "017069133835",
    "048819808253",
    "094389454867",
    "098760798382",
    "104374241257",
    "106583098589",
    "108575199400",
    "121021644041",
    "122526803553",
    "122578899357",
    "124985052026",
    "126357580389",
    "131013547314",
    "156813124566",
    "190319476487",
    "204538143572",
    "205585389593",
    "211330385671",
    "217643126080",
    "237065988967",
    "245545462676",
    "263625296855",
    "272398656194",
    "281123927165",
    "303241398832",
    "341280168497",
    "341593696636",
    "354813040037",
    "364406365360",
    "366743142698",
    "377024640650",
    "380420809688",
    "411392592546",
    "414596584902",
    "437450045455",
    "440796970383",
    "442386744353",
    "446045086412",
    "450853457545",
    "451049120500",
    "452307495513",
    "453000072557",
    "453252182341",
    "454834333376",
    "457447274322",
    "467912361380",
    "468650794304",
    "492215442770",
    "503227376785",
    "510948584623",
    "520018980103",
    "536280801234",
    "539772159869",
    "550225433462",
    "563025443158",
    "574779866223",
    "579539705040",
    "588750061953",
    "590183813437",
    "590183933784",
    "598674086554",
    "607024016150",
    "626614931356",
    "628508329040",
    "633930458069",
    "637423239942",
    "638885417683",
    "651117190479",
    "659782779980",
    "662702820516",
    "669540362728",
    "675030665977",
    "680080141114",
    "680994064768",
    "692866216735",
    "705930551576",
    "709848358524",
    "720646828776",
    "727897471807",
    "730335477804",
    "737474898029",
    "740489534195",
    "746614075791",
    "749857270468",
    "763603941244",
    "764974769150",
    "772153158452",
    "777275614652",
    "780543022126",
    "783357654285",
    "801668240914",
    "811711786498",
    "833128469047",
    "834264404009",
    "835444307964",
    "843974653677",
    "853188333426",
    "867004704886",
    "871362719292",
    "875698925577",
    "886529160074",
    "890145073186",
    "895015795356",
    "898809789911",
    "907027046896",
    "912233562940",
    "914824155844",
    "933208885752",
    "951798379941",
    "975050140332",
    "978288397137",
    "990339680094",
}

# Exact public Hub commit pins reviewed against the original model API.
# These are complete integrity identifiers, never account-ID exemptions.
PUBLIC_MODEL_REVISION_PINS = {
    # https://huggingface.co/api/models/Qwen/Qwen3.5-9B/revision/c202236235762e1c871ad0ccb60c8ee5ba337b9a
    "c202236235762e1c871ad0ccb60c8ee5ba337b9a",  # pragma: allowlist secret
}

# Exact public HumanEval metadata hashes, recomputed from all164 official cases.
# Original public source revision: 6d43fb980f9fee3c892a914eda09951f772ad10d.
# The214438-byte JSONL has SHA256:
# 1d49078ba3e2b196b9344535bef34a43021f038fad9561d6ee7c53450609a6a2
# Only these21 complete hashes contain account-shaped fragments; no path or
# general hash exemption is granted. Private/secret scans still use original bytes.
PUBLIC_ORACLE_INTEGRITY_PINS = {
    "016242643122db3e21319d9a28d85c518e4dee3ec44846c783df43a8ba516206",  # pragma: allowlist secret
    "1c4358fa49db2bdf8ce6f24c604765987914a9f6d51a7109c68272c9f7f66cd8",  # pragma: allowlist secret
    "1eb46648867a6e499ee7e4fa6500b594937f325209204606d7391c8cad9df9c4",  # pragma: allowlist secret
    "33902aa28a76a3d18738481f56efa1dbf4a361328768356c6de78461113bfeee",  # pragma: allowlist secret
    "58845cc488ac0df5ed6876e026532380238c02e3035897db32428fa719bd4e38",  # pragma: allowlist secret
    "5de2b4ef399da58cdb28f9c2378ffa86b41dc20d205428828151cecea0d2968d",  # pragma: allowlist secret
    "66607b421ed8b5eb91de52ca96f1b071ecc536edf716451650816ae4e7701f64",  # pragma: allowlist secret
    "76075722f14f440573576993f00d079bbb312818c40a47974f99bf0d9374d905",  # pragma: allowlist secret
    "856a164439599802d5210e2969c1c5673c84b83b4bdca5db34384d7b10d3d741",  # pragma: allowlist secret
    "92ee00c52b7151e6d841ebb4bddc0cef994b5b0d65d7cf090033676450a203e9",  # pragma: allowlist secret
    "9a8b1ce14654166499b901763884822e6ed09c454f249a165f0c13d8027bed53",  # pragma: allowlist secret
    "9f7b61f73bfc14d91aa4c0ce07643312ead6911e631063872699b0409b924a2c",  # pragma: allowlist secret
    "a6e0d9cbd980c75fb58baac28d3f6b222e2d5bbbe2751b23d508010097921e46",  # pragma: allowlist secret
    "af63ce924260924735c5c424f29e2ca760971f45b9b1d63e1aef1a2b9b4bb26a",  # pragma: allowlist secret
    "b399707294183c0582701aa5a4ae4f0d5bee2550649c193cb9fd3f3c42b6a6b8",  # pragma: allowlist secret
    "c4f2fe113796217362ebfa8437244422d6e2ed83b00a97073c776a92bce55255",  # pragma: allowlist secret
    "ca340d7a433f9d538412794976c3323fd6efc900f8e22781e06c7e4d45942146",  # pragma: allowlist secret
    "e49218abe89b2b138512731659d96115b045637b8cebe43d2407656209332e58",  # pragma: allowlist secret
    "e85b90468fd66c094cdc9a39b501879444441c741ee3a9d954f3a3bc2c6b82c7",  # pragma: allowlist secret
    "f744311bc2c4ea109d693435515847f3f8292c154f780205ce2a614acc1890e0",  # pragma: allowlist secret
    "f8c8a5b02862a0d5acb032bb0e4b9d5b1f057077479202d96f7fb5eedf2db2e9",  # pragma: allowlist secret
}

# Exact synthetic request checksum independently recomputed from source-owned
# reload/training fixtures. Only this complete integrity token is exempted;
# numeric fragments, other checksums and original private/secret scans are not.
PUBLIC_FIXTURE_INTEGRITY_PINS = {
    "80aed81a4098edb131525060103bbff66b214a8ca82ed87d4d6b55b66916ea6b",  # pragma: allowlist secret
}

# Keep organization-specific hosts, groups and identities in an external release denylist.
INTERNAL_PATTERNS = (
    (
        "internal-host",
        re.compile(r"(?i)\b(?:[a-z0-9-]+[.])*(?:corp|internal)[.][a-z0-9.-]+"),
    ),
    (
        "personal-path",
        re.compile(
            r"/" + r"Users/[^/\s]+|"
            r"(?<!~)/" + r"home/(?!user(?:/|\b)|appuser(?:/|\b))[^/\s]+"
        ),
    ),
)

SECRET_PATTERNS = (
    re.compile(
        r"(?:AKIA|ASIA|A3T[A-Z0-9]|AGPA|AIDA|AROA|AIPA|ANPA|ANVA)"
        r"[A-Z0-9]{12,}"
    ),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"glpat-[A-Za-z0-9_-]{12,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"tvly-[A-Za-z0-9_-]{8,}"),
    re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
)

FORBIDDEN_TOP_LEVEL = {
    "artifacts",
    "checkpoints",
    "data",
    "datasets",
    "models",
    "papers",
}

FORBIDDEN_SUFFIXES = {
    ".arrow",
    ".ckpt",
    ".csv",
    ".db",
    ".docx",
    ".feather",
    ".gif",
    ".gz",
    ".h5",
    ".hdf5",
    ".ipynb",
    ".joblib",
    ".jpeg",
    ".jpg",
    ".log",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pdf",
    ".pickle",
    ".pkl",
    ".pmml",
    ".png",
    ".pt",
    ".pth",
    ".safetensors",
    ".sqlite",
    ".tflite",
    ".tsv",
    ".webp",
    ".whl",
    ".xlsx",
    ".zip",
}

FORBIDDEN_PATH_PARTS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "cdk.out",
    "dist",
    "output",
    "runs",
    "tmp",
}

REQUIRED_PUBLIC_PATHS = {
    ".github/workflows/public-release.yml",
    ".gitlab-ci.yml",
    ".secrets.baseline",
    "AGENTS.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "scripts/verify_repository_governance.py",
    "tests/test_repository_governance.py",
    "tests/test_repository_policy.py",
}


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def scan_text(path: str, text: str, private_literals: Iterable[str] = ()) -> list[Finding]:
    findings: list[Finding] = []

    if any(value.casefold() in text.casefold() for value in private_literals if value):
        findings.append(Finding(path, "private-literal", "contains a private denylist value"))

    for code, pattern in INTERNAL_PATTERNS:
        if pattern.search(text):
            findings.append(Finding(path, code, "contains a private coordinate or identifier"))

    for pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0).lower()
            if re.fullmatch(r"tvly-x{8,}", value):
                continue
            findings.append(Finding(path, "secret-token", "contains a credential-shaped value"))
            break

    # Only validated integrity-digest fields and public package hash paths are exempt.
    # Comments, URLs and all other lockfile fields still receive account-ID checks.
    account_text = re.sub(
        r'(\bhash\s*=\s*")(?:sha256:[a-f0-9]{64}|sha384:[a-f0-9]{96}|sha512:[a-f0-9]{128})(?=")',
        r"\1<integrity-digest>",
        text,
        flags=re.IGNORECASE,
    )
    account_text = re.sub(
        r"https://files[.]pythonhosted[.]org/packages/[a-f0-9]{2}/[a-f0-9]{2}/[a-f0-9]{60}/",
        "<public-package-hash>/",
        account_text,
        flags=re.IGNORECASE,
    )
    account_text = re.sub(
        r"(?<!\S)--hash=sha256:[a-f0-9]{64}(?=\s|$)",
        "--hash=<integrity-digest>",
        account_text,
        flags=re.IGNORECASE,
    )
    for revision in PUBLIC_MODEL_REVISION_PINS:
        account_text = re.sub(
            rf"(?<![a-fA-F0-9]){re.escape(revision)}(?![a-fA-F0-9])",
            "<reviewed-public-model-revision>",
            account_text,
        )
    for digest in PUBLIC_ORACLE_INTEGRITY_PINS | PUBLIC_FIXTURE_INTEGRITY_PINS:
        account_text = re.sub(
            rf"(?<![a-fA-F0-9]){re.escape(digest)}(?![a-fA-F0-9])",
            "<reviewed-public-oracle-integrity>",
            account_text,
        )
    for account_id in re.findall(r"(?<!\d)\d{12}(?!\d)", account_text):
        if account_id not in PUBLIC_AWS_ACCOUNT_IDS:
            findings.append(Finding(path, "unknown-account-id", "contains an unreviewed 12-digit value"))

    return findings


def scan_path(path: str) -> list[Finding]:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    parts = tuple(part for part in normalized.split("/") if part)
    findings: list[Finding] = []

    if normalized.casefold() in {".github/dependabot.yml", ".github/dependabot.yaml"}:
        findings.append(
            Finding(normalized, "public-write-automation", "automatic public PRs bypass approval")
        )

    if parts and parts[0].lower() in FORBIDDEN_TOP_LEVEL:
        findings.append(Finding(normalized, "internal-or-generated-path", "forbidden top-level path"))

    if any(part.lower() in FORBIDDEN_PATH_PARTS for part in parts):
        findings.append(Finding(normalized, "generated-artifact", "generated path is tracked"))

    if Path(normalized).suffix.lower() in FORBIDDEN_SUFFIXES:
        findings.append(Finding(normalized, "generated-artifact", "generated file type is tracked"))

    return findings


def missing_required_paths(repo: Path, required: Iterable[str]) -> list[str]:
    return sorted(path for path in required if not (repo / path).is_file())


def tracked_paths(repo: Path) -> list[str]:
    output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return [item.decode("utf-8") for item in output.split(b"\0") if item]


def _candidate_entries(repo: Path, ref: str | None) -> list[tuple[str, str, str]]:
    if ref is None:
        args = ["ls-files", "--stage", "-z"]
    else:
        tree = _git(repo, "rev-parse", "--verify", f"{ref}^{{tree}}")
        args = ["ls-tree", "-r", "-z", tree]
    raw = subprocess.check_output(["git", *args], cwd=repo)
    entries = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, path = record.split(b"\t", 1)
        fields = metadata.decode("ascii").split()
        mode = fields[0]
        oid = fields[1] if ref is None else fields[2]
        if ref is None and fields[2] != "0":
            mode = "unmerged"
        entries.append((mode, oid, path.decode("utf-8")))
    return entries


def scan_repository(repo: Path, ref: str | None = None) -> list[Finding]:
    """Scan immutable index or commit blobs, never working-copy or symlink target bytes."""
    findings: list[Finding] = []
    denylist_path = os.environ.get("AMLAS_RELEASE_DENY_LITERALS_FILE")
    private_literals = []
    if denylist_path:
        private_literals = [
            line.strip()
            for line in Path(denylist_path).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    entries = _candidate_entries(repo, ref)
    paths = {path for _, _, path in entries}
    for path in sorted(REQUIRED_PUBLIC_PATHS - paths):
        findings.append(Finding(path, "required-path-missing", "public governance file is absent"))
    for entry_number, (mode, oid, relative) in enumerate(entries, 1):
        path_findings = scan_text("", relative, private_literals)
        display_path = f"<redacted-path-{entry_number}>" if path_findings else relative
        findings.extend(
            Finding(display_path, item.code, "candidate path contains a prohibited identifier")
            for item in path_findings
        )
        findings.extend(item._replace(path=display_path) for item in scan_path(relative))
        exact_dox_alias = (
            mode == "120000"
            and relative == "CLAUDE.md"
            and any(
                target_mode == "100644" and target_path == "AGENTS.md"
                for target_mode, _, target_path in entries
            )
            and subprocess.check_output(["git", "cat-file", "blob", oid], cwd=repo) == b"AGENTS.md"
        )
        if mode not in {"100644", "100755"} and not exact_dox_alias:
            findings.append(
                Finding(display_path, "unsupported-git-mode", "symlink, submodule or unresolved entry")
            )
            continue
        size = int(_git(repo, "cat-file", "-s", oid))
        if size > 10 * 1024 * 1024:
            findings.append(Finding(display_path, "large-file", "tracked blob exceeds 10 MiB"))
            continue
        raw = subprocess.check_output(["git", "cat-file", "blob", oid], cwd=repo)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            findings.append(Finding(display_path, "binary-file", "tracked blob is not UTF-8 text"))
            continue
        findings.extend(scan_text(display_path, text, private_literals))
    return findings


def refs_have_equal_trees(repo: Path, left: str, right: str) -> bool:
    return _git(repo, "rev-parse", f"{left}^{{tree}}") == _git(repo, "rev-parse", f"{right}^{{tree}}")


def _print_findings(findings: Iterable[Finding]) -> None:
    for finding in findings:
        print(f"{finding.code}: {finding.path}: {finding.detail}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--scan-tree", action="store_true")
    parser.add_argument("--ref", help="Scan this immutable commit/tree; defaults to the staged index")
    parser.add_argument("--compare-refs", nargs=2, metavar=("GITLAB_REF", "GITHUB_REF"))
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    failed = False

    if args.scan_tree or not args.compare_refs:
        findings = scan_repository(repo, args.ref)
        if findings:
            _print_findings(findings)
            failed = True
        else:
            print(f"public_tree_scan=pass tracked_files={len(_candidate_entries(repo, args.ref))}")

    if args.compare_refs:
        left, right = args.compare_refs
        if refs_have_equal_trees(repo, left, right):
            tree = _git(repo, "rev-parse", f"{left}^{{tree}}")
            print(f"main_tree_sync=pass tree={tree}")
        else:
            left_tree = _git(repo, "rev-parse", f"{left}^{{tree}}")
            right_tree = _git(repo, "rev-parse", f"{right}^{{tree}}")
            print(
                f"main_tree_sync=fail {left}={left_tree} {right}={right_tree}",
                file=sys.stderr,
            )
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
