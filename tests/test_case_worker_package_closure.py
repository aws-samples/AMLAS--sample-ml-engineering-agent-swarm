"""Exact data-only image package cannot import agent/SDK/application state."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_fixed_package_has_only_data_modules_and_no_application_imports(tmp_path):
    root = Path(__file__).resolve().parents[1]
    package = tmp_path / "case_worker_pkg"
    names = (
        "backends/types.py",
        "data_eng/coding_case_profile.py",
        "data_eng/coding_case_codec.py",
        "data_eng/coding_case_manifest.py",
        "data_eng/coding_case_managed.py",
        "data_eng/inspection_schema.py",
        "sagemaker_entrypoints/coding_case_freeze.py",
    )
    for name in names:
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / "src/amlas" / name).read_bytes())
        (target.parent / "__init__.py").touch()
    (package / "__init__.py").touch()
    code = """import importlib.abc,sys
sys.path.insert(0,sys.argv[1])
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'amlas','boto3','botocore','strands','torch','transformers','peft'}: raise RuntimeError('forbidden image dependency '+fullname)
sys.meta_path.insert(0,Deny())
from case_worker_pkg.sagemaker_entrypoints.coding_case_freeze import _source
from case_worker_pkg.data_eng.coding_case_managed import _expected
from case_worker_pkg.backends.types import VerifiedPreparationSource
print('CASE-PACKAGE-CLOSED')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(tmp_path)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "CASE-PACKAGE-CLOSED"
