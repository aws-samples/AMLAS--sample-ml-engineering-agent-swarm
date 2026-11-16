"""Package source links remain closed; source bytes and dependencies are unchanged."""

import os
import tarfile
import tomllib
from pathlib import Path

import pytest


def test_root_dox_link_is_excluded_together_with_its_target():
    project=tomllib.loads(Path('pyproject.toml').read_text())
    exclusions=project['tool']['hatch']['build']['targets']['sdist']['exclude']
    assert '/AGENTS.md' in exclusions
    assert '/CLAUDE.md' in exclusions, 'sdist excludes root target but includes dangling CLAUDE link'
    assert '/tests' in exclusions and '/processing_bootstrap/tests' not in exclusions


def test_built_sdist_has_no_dangling_root_dox_link():
    artifact=os.environ.get('AMLAS_SDIST_INSPECTION_ARTIFACT')
    if artifact is None:
        pytest.skip("actual built sdist not supplied; declaration control still runs")
    with tarfile.open(artifact,'r:gz') as archive:
        names=set(archive.getnames())
        for member in archive.getmembers():
            if member.issym():
                assert str(Path(member.name).parent/member.linkname) in names, 'dangling sdist link: '+member.name
