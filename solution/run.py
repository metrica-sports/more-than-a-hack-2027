"""Regenerate the submission file from the provided data.

    python solution/run.py --out evaluation/results/<team_name>.csv
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from possession import pipeline  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "evaluation", "results", "hmm_v1.csv"))
    args = ap.parse_args()
    res = pipeline.run()
    print("written", pipeline.write_submission(res["labels"], args.out))
