---
layout: site
title: Markov Chain Monte Carlo
permlink: /notes/
exclude: true
---

Main reference:
- PRML book chapter 11
- [A Conceptual Introduction to Hamiltonian Monte Carlo](https://arxiv.org/pdf/1701.02434.pdf)

### Phase Space
Define joint probability over phase space $(q, p)$, where $q$ is the target distribution (state) and $p$ is the auxiliary distribution (momentum) introduced, as $\pi(q, p)$. Write it in terms of an invariant *Hamiltonian* function, $H(q, p)$ such that:
$$
\pi(q,p)=\exp(-H(q, p))
$$

which decomposes into two terms:

$$
\begin{aligned}
H(q,p) &= -\log \pi(p|q)-\log\pi(q)\\
&= K(p, q) + V(q)
\end{aligned}
$$

where $K(p,q)$ is *kinetic energy* and $V(q)$ is *potential energy* and the value of Hamiltonian $H(q,p)$ itself in the phase space is called the *energy* at that point.


### Hamiltonian Equations

$$
\begin{aligned}
\frac{\mathrm{d}q}{\mathrm{d}t} &= +\frac{\partial H}{\partial p}=\frac{\partial K}{\partial p}\\
\frac{\mathrm{d}p}{\mathrm{d}t} &=-\frac{\partial H}{\partial q}=-\frac{\partial K}{\partial q}-\frac{\partial V}{\partial q}
\end{aligned}
$$

### Invariance
Hamiltonian preserves volume in phase space:

Consider the vector filed $V=(\frac{dq}{dt}, \frac{dp}{dt})$ and its divergence:

$$
\begin{aligned}
\mathrm{div} V &= \frac{\partial}{\partial q} \frac{dq}{dt} + \frac{\partial}{\partial p}\frac{dp}{dt}\\
&=\frac{\partial}{\partial q}\frac{\partial H}{\partial p} - \frac{\partial}{\partial p}\frac{\partial H}{\partial q}\\
&= 0
\end{aligned}
$$

since the divergence vanishes, the volume of the phase space is a constant.


Hamiltonian itself is also constant over time:

$$
\begin{aligned}
H(q+\delta q, p+\delta p)
&= H(q, p) + H'_q \delta q + H'_p \delta p + o(\delta t^2)\\
&= H(q, p) + H'_qH'_p\delta t-H'_p H'_q \delta t + o(\delta t^2)\\
&= H(q, p) + o(\delta t^2)
\end{aligned}
$$


### Geometry of Phase Space
Since the value of Hamiltonian is constant, every Hamiltonian trajectory is confined to an energy level set:

$$
H^{-1}=\{q, p| H(q, p)=E\}.
$$

For every point in phase space, we can uniquely determine it by first finding the proper energy level set and then find its location within that energy level set.

$$
\pi(q,p)=\pi(\theta_E|E)\pi(E)
$$

which is called *microcannonical decomposition*. The conditional distribution over each energy level set $\pi (\theta_E|E)$ is called *microcanonical distribution* and the distribution over all the level sets $\pi(E)$ is called *marginal energy distribution*.

#### Two phases of Hamiltonian Markov chain
- *Determinstic exploration*: within a given energy level set, the Hamiltonian trajectory is integrated deterministically. The efficacy is determined by how long the trajectory can be integrated.
- *Stochastic exploration*: Since Hamilton's equations preserve the value of Hamiltonian, one has to randomly *jump* to different energy levels to explore the entire phase space. The performance of the stochastic explorration is determined by how quickly the random walk can diffuse across the energies typical to the marginal energy distribution $\pi(E)$. Intuitively, the energy transition procedure is controlled by  $\pi(E|q)$ , if $\pi(E|q)$ is narrow compared to the marginal energy distribution $\pi(E)$, it can be very slow to explore the typical set of distribution $\pi(E)$.

#### Choice of Kinetic Energy
- Euclidean-Gaussian Kinetic Energies: $\pi(p|q)=\mathcal{N}(p|0, M)$ where $M=R\cdot S\cdot g\cdot S^\top \cdot R\top$ is a generalized covariance matrix; $g$ is covariance matrix; $R$ is orthogonal rotation matrix; $S$ is diagonal scaling matrix. Thus the kinetic energy is $K(q,p)=\frac{1}{2}p^\top M^{-1} p+\log |M| + \mathrm{const.}$
- Riemannian-Gaussian Kinetic Energy: $\pi(p|q)=\mathcal{N}(p|0, \Sigma(q))$ where covariance $\Sigma(q)$ is a function of state $q$.

#### Choice of Integration Time
If we intergrate the Hamiltonian trajectory long enough, its temporal expectation approximates its spatial expectation. However, the return is diminishing while integration time grows. We need to pick a proper integration time which is not too long but makes temporal expectation approximates spatial expectation with enough precision.

Near-optimal termination criterion: No-U-Turn.
