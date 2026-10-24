#!/bin/sh
# Publish docs/ to the public site repository BoltzmannEntropy/scqfig.github.io.
# The code repository stays private; only the built site goes public.
# Usage: sh tools/publish_site.sh ["commit message"]
set -eu

SITE_REPO=git@github.com:BoltzmannEntropy/scqfig.github.io.git
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MSG=${1:-"Publish site from scqfig $(git -C "$ROOT" rev-parse --short HEAD)"}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

git clone -q "$SITE_REPO" "$WORK/site"
# Replace everything except .git with the current docs/ (internal review notes stay out).
find "$WORK/site" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
rsync -a --exclude review/ --exclude .DS_Store "$ROOT/docs/" "$WORK/site/"

cd "$WORK/site"
git add -A
if git diff --cached --quiet; then
  echo "Site already up to date."
  exit 0
fi
git -c user.name=BoltzmannEntropy \
    -c user.email=91342039+BoltzmannEntropy@users.noreply.github.com \
    commit -q -m "$MSG"
git push -q origin HEAD:main
echo "Published: https://boltzmannentropy.github.io/scqfig.github.io/"
