"""Regenerate the submission from the data bundle:

    python run_pipeline.py --team MyTeam        # writes submissions/MyTeam.csv, then validates

The data is read from ./data or $MTAH_DATA_DIR (see DATA.md).
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team", required=True, help="team name; names the output file")
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "submissions"))
    args = parser.parse_args()

    from possession.pipeline import run

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, args.team + ".csv")
    run(out)
    print("wrote", out)
    sys.exit(subprocess.call([sys.executable,
                              os.path.join(ROOT, "evaluation", "validate_submission.py"), out]))
