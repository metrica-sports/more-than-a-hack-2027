# CLAUDE.md

## Dependencies

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml`; `uv.lock` is
not committed).
The requirements files stay in the repository so participants can still install with `pip`.

**Rule: any dependency added with uv must also be added to the requirements files, in the same
change.**

| uv command                   | Also add it to             |
| ---------------------------- | -------------------------- |
| `uv add <pkg>`               | `requirements.txt`         |
| `uv add --group starter <pkg>` | `requirements-starter.txt` |
