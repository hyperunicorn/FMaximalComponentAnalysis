# FMaximalComponentAnalysis

### The Theory
A PCA-style algorithm for maximising the ratio of variances of a two class data-set
This package implements an algorithm I designed called F-Maximal Component Analysis or FMCA for short.
The basic idea is best understood in reference to its inspiration, Principal Component Analysis or PCA for short. In PCA, you try to find a basis vector that contains as much of the overall variance of the data set as possible. In FMCA, we pursue a similar but more complex strategy. Let $X_0,\ X_1 \in \mathbb{R}^n$ be vector-valued random variables such that 

$X_i = [X_{i1}, X_{i2}, ...,\ X_{in}]\ \forall\ i\ \in \\{ 1, 2 \\}$ and $Var(X_{ij})\ <\ \infty\ \forall\ (i,\ j)\ \in\ \\{ 1,2 \\} \times \\{ 1,...,n \\}$ 

It follows then that given a basis vector $v\ \in\ S^n$ where $S^n$ is the unit sphere in n-dimensions, the variance of the rotated component $Var(X_n \dot v) = \sum_{i=0}v_i^2 Var(X_{ni})\ +\ \sum_{j=0}\sum_{k \ne j}\ v_jv_kCov(X_{nj}, X_{nk})$.

The task then is to solve the extremisation problem: 

$argmax_{v\ \in\ S^n}\ \frac{Var(X_1\ \dot\ v)}{Var(X_0\ \dot\ v)}$

To solve this, let us first note that:

$Var(X_k\ \dot\ v)\ =\ v^T \Gamma_k\ v$

Where $\Gamma_k$ is the covariance matrix of $X_k$ and $v$ is a column vector.

Then it follows that we can reframe the extremisation problem as:

$argmax_{v\ \in\ S^n}\ \frac{v^T \Gamma_1 v}{v^T \Gamma_0 v}$

In other words, we have to optimise this ratio of quadratic forms. The task is to reframe this as an unconstrained quadratic optimisation problem, which has a well-understood solution.

First we should note that given that $v\ \in\ S^n$ it follows that $v^T\ \dot\ v\ =\ v\ \dot\ v^T\ =\ 1$ Therefore:

$v^T \Gamma_0 v \dot\ v^T \Gamma_{0}^{-1} v = 1$
Which implies that $(v^T \Gamma_0 v)^{-1} = v^T \Gamma_{0}^{-1} v$

Therefore we can restate our extremisation problem as:

$argmax_{v\ \in\ S^n}\ v^T \Gamma_1 v\ \dot\ v^T \Gamma_{0}^{-1} v\ =\ v^T \Gamma_1 \Gamma_{0}^{-1} v$

Let $Q = \Gamma_1 \Gamma_{0}^{-1}$

And let $Sym(Q) = \frac{Q\ +\ Q^T}{2}$ and $Skew(Q) = \frac{Q\ -\ Q^T}{2}$

Then $Q = Sym(Q)\ +\ Skew(Q)$
Then $v^T Q v\ =\ v^T Sym(Q) v\ +\ v^T Skew(Q) v$

But since $v^T Skew(Q) v\ =\ 0$

It follows that $v^T Q v\ =\ v^T Sym(Q) v$

Therefore $argmax_{v\ \in\ S^n}\ v^T Q v$ is simply the eigenvector corresponding to the largest eigenvalue of $Sym(Q)$.
