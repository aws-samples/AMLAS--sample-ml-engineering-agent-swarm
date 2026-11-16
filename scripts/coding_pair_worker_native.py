"""Fixed image-owned paired broker; guard OS identity and output before SDK imports."""

import importlib.util
import os
import stat
import sys

if __name__ == "__main__":
    source = os.environ.get("AMLAS_SOURCE_COMMIT")
    if source != os.environ.get("AMLAS_CODING_PAIR_SOURCE_COMMIT"):
        raise ValueError("fixed paired source selector differs")
    spec = importlib.util.spec_from_file_location("processing_identity", "/app/processing_identity.py")
    identity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(identity)
    # Common worker guards require no supplementary groups; the existing oracle
    # has the exact broker group10001 contract. Inspect it without changing shared guards.
    if (
        os.getgroups() != [10001]
        or os.getresuid() != (10001, 10001, 10001)
        or os.getresgid() != (10001, 10001, 10001)
    ):
        raise ValueError("fixed paired broker identity differs")
    values = identity._read_status()
    for field in ("Uid", "Gid"):
        if values.get(field, "").split() != ["10001"] * 4:
            raise ValueError("fixed paired broker filesystem identity differs")
    for field in ("CapEff", "CapPrm", "CapInh", "CapAmb"):
        value = values.get(field, "")
        if (
            not value
            or len(value) > 16
            or any(c not in "0123456789abcdefABCDEF" for c in value)
            or int(value, 16) != 0
        ):
            raise ValueError("fixed paired broker active capability differs")
    bounded = values.get("CapBnd", "")
    if (
        not bounded
        or len(bounded) > 16
        or any(c not in "0123456789abcdefABCDEF" for c in bounded)
        or int(bounded, 16) & ~identity.NATIVE_CEILING
    ):
        raise ValueError("fixed paired broker bounding ceiling differs")
    securebits = identity._prctl(27)
    if (
        values.get("NoNewPrivs") != "1"
        or values.get("Seccomp") != "2"
        or values.get("Threads") != "1"
        or identity._prctl(7) != 0
        or not 0 <= securebits <= 255
        or securebits & 0x14
    ):
        raise ValueError("fixed paired broker permanent privilege differs")
    output = os.fstat(3)
    named = os.stat("/opt/ml/processing/output", follow_symlinks=False)
    if (
        not stat.S_ISDIR(output.st_mode)
        or (output.st_uid, output.st_gid, stat.S_IMODE(output.st_mode)) != (10001, 10001, 0o700)
        or (output.st_dev, output.st_ino) != (named.st_dev, named.st_ino)
    ):
        raise ValueError("fixed paired original output descriptor differs")
    sys.path.insert(0, "/app/src")
    from amlas.sagemaker_entrypoints.coding_pair import main

    main()
