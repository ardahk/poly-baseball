#!/usr/bin/env bash
# Regenerate the README leaderboard from the live journal and push it if the
# standings actually moved. Safe to run unattended from a systemd timer: it only
# ever touches README.md, commits nothing else, and makes no commit at all when
# the numbers are unchanged (the updater leaves the file alone, timestamp
# included, so a quiet offseason produces zero commits).
#
# Auth: relies on `gh auth setup-git` having configured the gh credential
# helper for https://github.com (run once as the polybot user). No token is
# stored in this repo.
set -euo pipefail

REPO=/home/ubuntu/poly-baseball
PY=$REPO/.venv/bin/python
cd "$REPO"

# Never touch anything but README.md; bail if it is dirty for another reason.
if ! git diff --quiet -- README.md; then
    echo "README.md has uncommitted local changes; skipping to avoid clobbering." >&2
    exit 0
fi

"$PY" scripts/update_readme_stats.py

if git diff --quiet -- README.md; then
    echo "standings unchanged; no commit made."
    exit 0
fi

git add README.md
git commit -m "chore: daily leaderboard refresh" >/dev/null
git push origin HEAD
echo "leaderboard pushed."
