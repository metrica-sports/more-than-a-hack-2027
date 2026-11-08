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
    ap.add_argument("--params", default=None, help="percorso di params.json (default: solution/params.json)")
    ap.add_argument("--allow-defaults", action="store_true",
                    help="se params.json manca usa i default invece di fermarsi (solo per prove)")
    args = ap.parse_args()
    params, cfg = pipeline.load_params(args.params, required=not args.allow_defaults)
    res = pipeline.run(params, cfg)
    print("written", pipeline.write_submission(res["labels"], args.out))
