#!/usr/bin/env bash
# List all class methods decorated with @flashinfer_api, grouped by class,
# with full multi-line signatures preserved.
#
# Usage:
#   scripts/list_apis.sh [-n] [-p] [--ref REF] [path...]
#
# Options:
#   -n, --no-lines    Omit line numbers
#   -p, --no-paths    Omit file paths (implies -n; signatures-only output)
#   -r, --ref REF     Run against a git revision (tag/branch/sha) via temp worktree
#   -h, --help        Show this help
#
# Default path is flashinfer/
#
# Examples:
#   scripts/list_apis.sh --ref v0.6.9 -p
#   diff -u <(scripts/list_apis.sh --ref v0.6.9 -p) <(scripts/list_apis.sh -p)

set -euo pipefail

show_lines=1
show_paths=1
ref=""
paths=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--no-lines) show_lines=0; shift ;;
    -p|--no-paths) show_paths=0; show_lines=0; shift ;;
    -r|--ref) ref="$2"; shift 2 ;;
    -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --) shift; paths+=("$@"); break ;;
    -*) echo "unknown flag: $1" >&2; exit 2 ;;
    *) paths+=("$1"); shift ;;
  esac
done

if [[ -n "$ref" ]]; then
  repo_root=$(git rev-parse --show-toplevel)
  if ! git -C "$repo_root" rev-parse --verify --quiet "$ref^{commit}" >/dev/null; then
    for remote in upstream origin; do
      git -C "$repo_root" remote get-url "$remote" >/dev/null 2>&1 || continue
      echo "fetching $ref from $remote..." >&2
      if git -C "$repo_root" fetch --quiet "$remote" "tag" "$ref" 2>/dev/null \
         || git -C "$repo_root" fetch --quiet "$remote" "$ref" 2>/dev/null; then
        break
      fi
    done
    git -C "$repo_root" rev-parse --verify --quiet "$ref^{commit}" >/dev/null \
      || { echo "ref '$ref' not found locally or on remotes" >&2; exit 1; }
  fi
  wt=$(mktemp -d -t fi-apis-XXXXXX)
  trap 'git -C "$repo_root" worktree remove --force "$wt" >/dev/null 2>&1; rm -rf "$wt"' EXIT
  git -C "$repo_root" worktree add --detach --quiet "$wt" "$ref"
  [[ ${#paths[@]} -eq 0 ]] && paths=("flashinfer/")
  paths=("${paths[@]/#/$wt/}")
fi

[[ ${#paths[@]} -eq 0 ]] && paths=("flashinfer/")

rg -HUn -U \
   "^class \w+[^\n]*:|^\s*@flashinfer_api(?:\([^)]*\))?|^\s*def \w+\([\s\S]*?\) *(?:-> *[^:]+)?:" \
   "${paths[@]}" \
| awk -v show_lines="$show_lines" -v show_paths="$show_paths" -v strip="${wt:-}/" '
    function emit(line,    out) {
      out = line
      if (strip != "/" && index(out, strip) == 1) out = substr(out, length(strip) + 1)
      if (!show_lines && !show_paths) sub(/^[^:]+:[0-9]+:/, "", out)
      else if (!show_lines)           sub(/:[0-9]+:/, ":", out)
      else if (!show_paths)           sub(/^[^:]+:/, "", out)
      print out
    }
    function flush(    n, i, parts) {
      if (pending) {
        n = split(pending, parts, "\n")
        for (i = 1; i <= n; i++) emit(parts[i])
        pending = ""; in_def = 0
      }
    }
    {
      path = $0; sub(/:[0-9]+:.*/, "", path)
      content = $0; sub(/^[^:]+:[0-9]+:/, "", content)
      if (path != lastpath) { flush(); cls=""; deco=""; printed=0; lastpath=path }
      if (content ~ /^class /)                { flush(); cls=$0; printed=0; deco=""; next }
      if (content ~ /^[ \t]+@flashinfer_api/) { flush(); deco=$0; next }
      if (content ~ /^[ \t]+def /) {
        flush()
        if (deco != "" && cls != "") {
          if (!printed) { emit(cls); printed=1 }
          emit(deco)
          pending = $0; in_def = 1
        }
        deco = ""
        next
      }
      if (in_def) pending = pending "\n" $0
    }
    END { flush() }
'
