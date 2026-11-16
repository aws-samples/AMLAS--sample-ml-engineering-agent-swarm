"""Image-owned case wrapper; the static initializer already dropped privilege."""

import sys

sys.path.insert(0, "/app")
from case_worker_pkg.sagemaker_entrypoints.coding_case_freeze import main

raise SystemExit(main())
