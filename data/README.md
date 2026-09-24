# Unpack the match data here

This folder is empty on purpose. The match assets are gated — see [DATA.md](../DATA.md) for how to
enrol, accept the terms, and get the download link.

Unpack the bundle here, keeping the names it arrives with:

```
data/
├─ tactical/                                  the evaluation reference
│  ├─ BAR-ATH_tactical.mp4                    row i of a submission is frame i of this
│  ├─ BAR-ATH_tactical_atd.xml                tracking metadata
│  ├─ BAR-ATH_tactical_atd.txt                tracking, players and ball
│  ├─ BAR-ATH_tactical_homography.csv         one 3x3 matrix per frame
│  └─ BAR-ATH_tactical_smart_tagging.xml      638 tactical phases and set pieces
└─ broadcast/                                 optional, NOT synchronised to tactical
   ├─ BAR-ATH_broadcast.mp4
   ├─ BAR-ATH_broadcast_atd.xml
   ├─ BAR-ATH_broadcast_atd.txt
   └─ BAR-ATH_broadcast_homography.csv
```

One folder per recording, because the question that matters most about any of these files is which
timeline it is on. `games.py` derives every path, so nothing else in the repository spells one out:

```python
from games import asset_paths
asset_paths()                      # tactical, the default
asset_paths(source="broadcast")
```

If a filename ever changes, `games.py` is the only edit.

Prefer to keep the data somewhere else? Point `MTAH_DATA_DIR` at it and nothing else changes:

```bash
export MTAH_DATA_DIR=/mnt/big-disk/mtah2027       # macOS / Linux
$env:MTAH_DATA_DIR = "D:\mtah2027"                # Windows PowerShell
```

## Nothing in here is committed

`.gitignore` keeps everything except this README out of git. That is not only tidiness — the terms
you accepted cover redistribution, and a public fork with the match video in it is redistribution.
Your own submission repository has to be public too, so check the same thing there.

## You do not need any of this to run the scorer

The evaluation reads CSVs only. `python evaluation/run_evaluation.py` works on a fresh clone with
this folder empty, and scores the four baselines that ship with the repository.
