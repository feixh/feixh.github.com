---
layout: site
title: SLAM
permlink: /notes/
exclude: true
---

## Square Root SAM Simulatenous Localization and Mapping via Square Root Information Smoothing
Frank Dellaert & Michael Kaess
- useful references:
    1. square root algorithms:  Factorization methods for discrete sequential estimation
    2. sparse matrix and graph theory: Graph Theory and Sparse Matrix Computations
    3. G. H. Golub & C.F. Van Loan. matrix computations. JHU press.
- Three forms of graphical models are present to model the SLAM problem

    1. Belief Net: joint probability of poses, landmarks and observations
        $$
            P(X, L, Z)=P(x_0)\prod P(x_i|x_{i-1}, u_i) \prod P(z_k|x_{i_k}, l_{j_k})
        $$
    2. Factor Graph: measurements are treated as parameters of the joint probability factors over the actual unknowns (poses & landmarks). Only single and pairwise cliques are considered in the SLAM setup.
        $$
           P(\Theta)\propto \prod\Phi_i(\theta_i)\prod_{i< j}  \Psi_{ij}(\theta_i, \theta_j)
        $$
        where $\Phi_0(x_0) \propto P(x_0)$, $\Psi_{(i-1)i}(x_{i-1}, x_i) \propto P(x_i | x_{i-1}, u_i)$ and $\Psi_{i_k j_k}(x_{i_k}, l_{j_k}) \propto P(z_k|x_{i_k}, l_{j_k})$.
    3. Markov Random Field: undirected and not factor nodes (adjacency indicatres which variables are linked by a common factor -- measurement or constraint). The graph is quite similar to the factor graph but without factor nodes.
- Inference

    A maximum a posteriori (MAP) estimate is obtained by solving an non-linear least sequare problem, which, in practice, is usually solved by a series of linear equations. After linearization and a bunch of matrix manipulation, it boils down to solving the normal equations $A^\top A \delta = A^\top b$ where $A$ is a sparse matrix which contains jacobians of both motion and measuremnt process; $b$ is a stack of odometry prediction errors and measurement prediction errors.

    Normally this is solved by *Cholesky* or LDL factorization of the *information matrix* $I=A^\top A$ or QR decomposition of A.

- Matrices & Graphs

    Information matrix should be block-sparse by construction. Since $I_{i,j} = \sum_k A^\top_{i,k} A_{k,j}=\sum_k A_{k,i}A_{k,j}$. Imagine each block as a scalar, then row index of A indicates an odometry constraint or a measurement, each column index indicates a pose or a landmark. i, j cannot indicate landmarks in the same time since each row only describes one measurement equation and thus only involves only one landmark. i, j can be either two adjacent poses (a row in A corresponding an odometry constraint) or a landmark and a pose (a row in A corresponding a measurement equation), where the landmark is observed by that pose. Therefore *at the block-level the sparsity pattern of the information matrix is exactly the adjacency matrix of the associated MRF*.

    It has been shown that when marginalizing out the past trajectory, the information matrix becomes completely dense. In contrast, in the smoothing setup (instead of filtering) the information matrix never becomes dense, since the past states are never marginalized out.

    Why the information matrix becomes dense after marginalization?

- Factorization & Variable Elimination

    In any of the three factorization schemes (Cholesky, LDL and QR), every time a variable (pose or landmark variable, measurements NOT unknowns) is eliminated, new edges should be added to the graph such that all the variables originally have connections to the eliminated variable should now connect to each other. This introduces many new edges to the graph, and, equivalently, adds more non-zero entries in the matrix R (destorys sparsity).

    Details about the equivalence on matrix factorization and variable elimination in graph?

- Improve performance & reduce fill-in

    ```The single most important factor to good performance is the order in which variables are eilimnated*. Different variable orderings can yield dramatically more or less fill-in, defined as the amount of edges added into the graph during factorization.```

    Popular method for variable elimination include colamd and generalized nested dissection.

    The paper actually introduces a *trik* for efficient variable elimination: run colamd on the *sparsity pattern* (where we treat each block as a scalar or a dot in the plot of a  matrix) of the blocks instead of passing it the original measurement jacobian A. This yields a performance improvement by a factor of 2 to sometimes 100.

    This general efficient factorization scheme has connections to the widely known Schur complement approach (partitioned inverse) in structure from motion. (can be found in the famous bundle adjustment survey and the large-scale SfM paper from google, cannot remember the authors thougth)

---
## Simultaneous Localization and Mapping with Sparse Extended Information Filter
Sebastian Thrun, Yufeng Liu, Daphne Koller, Andrew Y. Ng, Zoubin Ghahramani, Hugh Durrant-Whyte

---
## Visual SLAM: Why Filter?
Hauker Strasdat, J.M.M. Montiel, Andrew J. Davison

- Fig 1. demonstrates Markov Random Field of filtering based and sparse keyframe based SLAM. Filtering based approach marginalized out past poses and added connections between landmarks which have been observed by the marginalized poses, thus lost sparsity. BA based approach selected some keyframess based on heuristics and thus discarded information, however sparsity of the problem has been kept.

- The paper mentioned that cost of BA is linear in N where N is the number of features while filtering is cubic in N. **WHY?**

- In an EKF based approach, past poses are naturally marginalized out, but how about features? Drop the features when they are out of view or what? This is NOT done right in Corvis.

- *Gauss-Newton* filter: algebraically equivalent to the Square Root Information Filter (SRIF), which never constructs the normal equations explicitly and solves the problem using an orthogonal decomposition on the square root form. -- the comments on the SRIF can be useful at some point, need to read references:

    1. (Gauss-Newton filter) Monocular SLAM as a graph of coalesced observations. ES. Eade & T. Drummond.
    2. (Gauss-Newton filter) Bias reduction filter convergence for long range stereo. G. Sibley, L. Matthies, and G. Sukhatme.
    3. (SRIF) Extension of square-root filtering to include process noise. P. Dyer and S. McReynolds.
    4. (sparse information filter) Exactly sparse delayed state filters. J. J. Leonard.

- Feature parametrization: people have the highly similar parametrization approach and can have a paper in BMVC?

    Efficient feature parametrization for visual SLAM using inverse depth bundles. BMVC, 2008.

- Experiments of the paper: 4 different setups to simulate real world SLAM scenarios
    1. move sideaways, features are visible across all frames
    2. move sideaways, features are partially visible across frames
    3. move sideaways with rotation
    4. sharp forward turn -- rotation during forward motion

- Main observation of the paper: increasing the number of features leads to a significant entropy reduction (which is the criteria they came up with, can just think this as accuracy), while increasing the number of intermediate features has only a minor influence.

- Weakness of the paper: to make evaluation "fair", they use the same set of sparse frames for both BA and filtering, which is NOT fair to filtering -- filtering leverages on online processing of EVERY incoming datum, which is totally missed in this paper.


A note on Information Filter:
- From the p.d.f. in the form of mean $\mu$ and covariance $P$, we obtain the "information form" of the distribution:

$$
\begin{aligned}
p(x) &\propto \exp(-(x-\mu)^\top P^{-1}(x-\mu))\\
&= \exp(x^\top P^{-1}x  - 2(P^{-1}\mu)^\top x + \mu^\top P^{-1} \mu )\\
&\propto \exp(x^\top P^{-1}x - 2(P^{-1}\mu)^\top x)
\end{aligned}
$$
Drop the last term, since it's a constant. Now let $\Lambda=P^{-1}, b=P^{-1}\mu=\Lambda\mu$ we have the information form of the p.d.f.:
$$
p(x) \propto \exp(x^\top\Lambda x - 2b^\top x)
$$

The problem is when we need the state $x$, we have to solve $x=\Lambda^{-1}b$ which is expensive. However, int the AR/VR setup, what we need is not the full state (motion and structure), but only the motion part of the state, which can be low dimensional and **matrix inversion lemma** can be leveraged to efficiently get the motion state (??? -- Sebstian Thrun' paper is about this I think).

## A Sliding Window Filter for SLAM
Gabe Sibley

WONDERFUL PAPER! VERY CLEAR ILLUSTRATIONS OF SPARSITY PATTERN OF THE SYSTEM/INFORMATION MATRIX AND HOW IT ENVOLVES AS POSES/LANDMARKS ARE MARGINALIZED.

## Simultaneous Localization & Mapping with Sparse Extended Information Filter
Sebastian Thrun, et al.

Also present in Thrun's book. Really need to read this paper.

## Factorization Methods for Discrete Sequential Estimation
Gerald J. Bierman

Good book on square root algorithms and also cover numeric aspects of SRIF, KF and U-D. Need to read the latter half of the book.
