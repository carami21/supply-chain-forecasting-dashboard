"""Run every pipeline stage in order.

  etl -> aggregate -> forecast -> products -> anomalies -> inventory
      -> hm_styles (skipped if H&M raw data is absent) -> build_extract

Run from the repo root: python run_pipeline.py
Then: streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import aggregate  # noqa: E402
import anomalies  # noqa: E402
import build_extract  # noqa: E402
import etl  # noqa: E402
import forecast  # noqa: E402
import hm_styles  # noqa: E402
import inventory  # noqa: E402
import products  # noqa: E402

STAGES = [
    ("etl", etl.run),
    ("aggregate", aggregate.run),
    ("forecast", forecast.run),
    ("products", products.run),
    ("anomalies", anomalies.run),
    ("inventory", inventory.run),
    ("hm_styles", hm_styles.run),
    ("build_extract", build_extract.run),
]


def main() -> None:
    import os
    os.chdir(ROOT)  # stages use repo-relative paths
    start = time.time()
    for name, fn in STAGES:
        t = time.time()
        print(f"\n== {name} ==")
        fn()
        print(f"   ({time.time() - t:.1f}s)")
    print(f"\npipeline complete in {time.time() - start:.0f}s — "
          "streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
