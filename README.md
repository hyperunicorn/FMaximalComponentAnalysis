# Stat-Maximal Matrices
## Introduction
This package is a collection of algorithms I developed for finding basis vectors that optimise certain test statistics. So far I have implemented an algorithm for maximising the ratio of variances in a two-class data-set (FMCA) and one for optimising the variance adjusted distance between the means of two data-sets on the same support (WMCA) though I plan to add support for some other statistics in time.
The basic idea is best understood in reference to its inspiration, Principal Component Analysis or PCA for short. In PCA, you take the covariance matrix and find the eigendecomposition of this matrix. The eigenvector associated with the largest such eigenvalue will be the direction with the largest variance among all such directions. Stat-maximal matrices generalises this process to a larger class of test statistics in the context of two-class datasets. F-maximal Component Analysis (FMCA) finds the basis vector in which the ratio of variance of the two classes is maximised. Whereas Welch-Maximal Component Analysis finds the basis vector which maximises the squared distance between the means of the two classes adjusted for distance. The primary purpose for doing so is, as in PCA, dimensionality reduction and cluster analysis.

## Dependencies
The only dependencies are Pandas and Numpy at any version which supports the core functionality. Though I tested this code with numpy version 1.24.2 and Pandas version 1.5.3 and python version 3.10.10

## F-Maximal Component Analysis (FMCA)

### The Practice
We can initalise an FMCA object with: 
`fmc = FMCA(n_components, cov_estimator, cov_params, max_pos, as_df)`

n_components: Determines the number of eigen-vectors output by transform. If set to 'auto' n_components will be set to the square root of the number of features of the input data.

cov_estimator: Determines the algorithm used to estimate the covariance matrix of the data. Any estimator passed must implment a fit function and posess a covariance_ attribute. If set to None, the standard maximum likelihood estimator will be used, but you shouldn't do this because the standard ML estimator is very bad.

cov_params: Use this to pass a dictionary of any additional parameters you may wish to pass the covariance estimator.

max_pos: Maximises the variance of the positive samples divided by the variance of the negative samples if true.

as_df: If true, passes the decomposed dataset as a df. Otherwise, passes it as a numpy matrix.

We can fit this object to a dataset with the standard sklearn idiom:
`fmc.fit(X, y)`
Where X is a 2-dimensional array and y is a 1-dimensional vector with exactly two unique values

Then we can obtain a dimensionally reduced matrix in the maximised components with:
`rotated = fmc.transform(X, y=None) `
Where X is the same above, if y is not None then y will be simply be appended to the resulting df/np in the same order that it was passed
You can also access the entire eigenspectrum with through the spec attribute
Which is an array of tuples of the form `(eigenvalue, eigenvector)`
### The Theory
In PCA, you try to find a basis vector that contains as much of the overall variance of the data set as possible. In FMCA, we pursue a similar but more complex strategy. Let $X_0,\ X_1 \in \mathbb{R}^n$ be vector-valued random variables such that 

$X_i = [X_{i1}, X_{i2}, ...,\ X_{in}]\ \forall\ i\ \in \\{ 1, 2 \\}$ and $Var(X_{ij})\ <\ \infty\ \forall\ (i,\ j)\ \in\ \\{ 1,2 \\} \times \\{ 1,...,n \\}$ 

It follows then that given a basis vector $v\ \in\ S^n$ where $S^n$ is the unit sphere in n-dimensions, the variance of the rotated component $Var(X_n \dot \quad v) = \sum_{i=0}v_i^2 Var(X_{ni})\ +\ \sum_{j=0}\sum_{k \ne j}\ v_jv_kCov(X_{nj}, X_{nk})$.

The task then is to solve the extremisation problem: 

$argmax_{v\ \in\ S^n}\ \frac{Var(X_1\ \dot \quad v)}{Var(X_0\ \dot \quad v)}$

To solve this, let us first note that:

$Var(X_k\ \dot \quad v)\ =\ v^T \Gamma_k\ v$

Where $\Gamma_k$ is the covariance matrix of $X_k$ and $v$ is a column vector.

Then it follows that we can reframe the extremisation problem as:

$argmax_{v\ \in\ S^n}\ \frac{v^T \Gamma_1 v}{v^T \Gamma_0 v}$

In other words, we have to optimise this ratio of quadratic forms. The task is to reframe this as an unconstrained quadratic optimisation problem, which has a well-understood solution.

First we should note that given that $v\ \in\ S^n$ it follows that $v^T\ \dot \quad v\ =\ v\ \dot \quad v^T\ =\ 1$ Therefore:

$v^T \Gamma_0 v \dot \quad v^T \Gamma_{0}^{-1} v = 1$
Which implies that $(v^T \Gamma_0 v)^{-1} = v^T \Gamma_{0}^{-1} v$

Therefore we can restate our extremisation problem as:

$argmax_{v\ \in\ S^n}\ v^T \Gamma_1 v\ \dot \quad v^T \Gamma_{0}^{-1} v\ =\ v^T \Gamma_1 \Gamma_{0}^{-1} v$

Let $Q = \Gamma_1 \Gamma_{0}^{-1}$

And let $Sym(Q) = \frac{Q\ +\ Q^T}{2}$ and $Skew(Q) = \frac{Q\ -\ Q^T}{2}$

Then $Q = Sym(Q)\ +\ Skew(Q)$
Then $v^T Q v\ =\ v^T Sym(Q) v\ +\ v^T Skew(Q) v$

But since $v^T Skew(Q) v\ =\ 0$

It follows that $v^T Q v\ =\ v^T Sym(Q) v$

Therefore $argmax_{v\ \in\ S^n}\ v^T Q v$ is simply the eigenvector corresponding to the largest eigenvalue of $Sym(Q)$.


### The Practice
