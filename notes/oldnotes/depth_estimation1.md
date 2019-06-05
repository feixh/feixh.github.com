---
layout: site
title: Depth Estimation 1
permlink: /notes/
exclude: true
---

## Semantic Multi-view Stereo: Jointly Estimating Objects and Voxels

Ali Osman Ulusoy, Michael J. Black, Andreas Geiger

- Based on previous MRF framework [34], which not exploit semantic priors
- Foundations of volumetric reconstruction based on photo-consistency[21], which, together with its probabilistic extensions lack a global probabilistic model -- hard to interpret the results.
- *Recent approaches phrase volumetric reconstruction as inference in an MRF where voxels along each pixel's line of sight are connected via high-order ray potentials.*


---

## Multi-view Supervision for Single-view Reconstruction via Differentiable Ray Consistency

Shubham Tulsiani, Tinghui Zhou, Alexei A. Efros, Jitendra Malik

- Introduction & Related work are useful, come back later
- Representation
  - Shape Representation: 3D shape representation is parametrized as occupancy (misnomer, actually the probability of being empty) probabilities of cells in a discretized 3D voxel grid, denoted by the variable $x$. Discretization of the 3D space not necessarily to be uniform -- as long as it's possible to trace rays across the voxel grid and compute intersections with cell boundaries.
  - Observation: treat various types of observations alike, including depth images, foreground masks, etc. Observation-camera pair $(O, C)$ -- observation $O$ is measured by camera $C$.
- View-consistency loss: $L(x; (O, C))$. Given the camera intrinsics, we can convert the constraint provided by each pixel of observation $O$ to a constraint of a specific ray $r$, which passes through the corresponding pixel. $L(x; (O, C))=\sum_r L_r(x)$. *The task for formulating the view consistency loss is simplified to defining a differentiable ray consistency loss $L_r(x)$.*
- Ray-tracing in a probabilistic occupancy grid:
  - Motivation: *The probabilistic occupancy model induces a distribution of events that can occur to ray $r$ and we can define $L_r(x)$ by seeing the incompatibility of these events with the available observation $o_r$.*
  - Ray termination events: random variable $z_r$ denotes the voxel in which the ray terminates.
  - $z_r=i$ iff the previous voxels in the path are all un-occupied and the $i$-th voxel is occupied, probability distribution of $z_r$:

  $$
  p(z_r=i)=
  \begin{cases}
  (1-x_i^r)\prod_{j=1}^{i-1}x_j^r, &i \le N_r\\
  \prod_{j=1}^{N_r}x_j^r, &i=N_r+1
  \end{cases}
  $$

  where event $z_r=N_r+1$ means the ray doesn't terminate -- ray goes into to empty space.
- Per-ray consistency loss:
  - Given depth observations:

  $$
  \psi_r^{\text{depth}}(i)=|d_i^r-d_r|
  $$

  where $d_i^r$ is the dpeth computed from the grid index $i$. Also we have an associated probability for $p(z_r=i)$ which is given above.
  - Given foreground mask:$s_r\in \{0, 1\}$ where 0 denotes object mask and 1 otherwise.

$$
\psi_r^{\text{mask}}(i)=
\begin{cases}
s_r, &i \le N_r\\
1-s_r, & i=N_r+1
\end{cases}
$$

- Ray consistency loss:

  $$
  \begin{aligned}
  L_r(x) &= \mathbb{E}_{z_r}[\psi_r(z_r)]\\
  &=\sum_{i=1}^{N_r+1}\psi_r(i)p(z_r=i)
  \end{aligned}
  $$

- Derivatives of loss w.r.t. CNN predictions $x$: see appendix.
- Further reading: [16, 26, 31] detailed reconstruction.
  - Hierarchical Surface Prediction for 3D Object Reconstruction
  - OctNetFusion
- TODO: study the code. (in lua, should be fun.)

---

## OctNet

- Conventional octree requires traversel of nodes all the way from the root to the level where the querying cell lives, which makes convolutional operation, which requires frequent accesses to neighboring cells, unaffordable. A hybrid grid-octree structure ([32]Real-time rendering and dynamic updating of 3-d volumetric data.) is not as memory-efficient partially addressed this issue by restricting the maximal depth of the octree.
- U-shaped network architecutre:
  - [2] segnet: A deep convolutional encoder-deconder architecture for image segmentation
  - [53] 3d u-net: Learning dense volumetric segmentation from sparse annotation

---

## Unsupervised Learning of Depth and Ego-Motion from Video

Tinghui Zhou

- [10, 14, 16] seem quite related. Differences: previous work requires relative camera motion during training, while this work produces camera motion as one of the outputs.
  - [10] DeepStereo: Learning to predict new views from the world's imagery.
  - [14] Unsupervised CNN for single view depth estimation: Geometry to the rescue.
  - [16] Unsupervised monocular depth estimation with left-right consistency.
- [14] requires pose supervision; [7] Eigen's single-view depth estimation paper: requires full depth supervision.
- concurrent work: SfM-Net: Learning of structure and motion from video. Differences: SfM-Net explicitly solves for object motion while this work discounts regions undergoing motion, occlusion and other factors via *explainability mask*.
- While a single-view depth CNN and a camera pose estimation CNN are jointly trained from unlabeled video sequences, the resulting depth model and the pose estimation model can be used independently during inference.
- Training samples: short image sequences of scenes captured by a moving camera; scene structure should be mostly rigid and static.
- View synthesis objective:

$$
  L_{vs}=\sum_s\sum_p|I_t(p)-\hat I_s(p)|
$$

  where $\hat I_s$ is the source view $I_s$ warped to the target coordinate frame based on predicted depth image $\hat D_t$ and camera motion $\hat T_{t\rightarrow s}$. Differentiable rendering module [8].
- Warping from target pixel location $p_t$ to source view $p_s$:

$$
p_s \sim K\hat T_{t\rightarrow s}\hat D_t(p_t) K^{-1}p_t
$$

where $K$ is camera intrinsic matrix.

- To handle cases where the assumptions of static scene, no-occlusion and Lambertian surfaces dont hold, train a *explainability prediction* network to mask out target-source pairs by the per-pixel soft masks $\hat E_s$. Weighted view synthesis loss:

$$
L_{vs}=\sum_{<I_1, \cdots, I_N>} \sum_p \hat E_s(p)|I_t(p)-\hat I_s(p)|
$$

where trivial solution exists which is $\hat E_s$ always being zero. Introduce regularization term $L_{reg}(\hat E_s)$ that *encourages nonzero predictions by minimizing the cross-entropy loss with constant label 1 at each pixel location*.

- Gradient locality: gradients are mainly derived from the pixel intensity difference, which will be problematic if
  - the correct projection is located in a low-texture region
  - the current estimate is far away from the correct one
- Two ways to handle gradient locality:
  - use convolutional encoder-decoder archticture with a small bottleneck for the depth network that implicitly constrains the output to be globally smooth and facilitates gradients to propagate from meaningful regions to nearby regions.
  - explicit multi-scale and smoothness loss [14, 16] that allows gradients to be derived
- Single-view depth prediction as a module in the whole system is adopted from DispNet[35]. Also look at [47] DeMoN.
- **Camera motion estimation outperforms ORB-SLAM running on short sequences without relocalization & loop closure, especially in the case of forward motion -- quite intrigering result!!!**

- [PyTorch implementation]( https://github.com/ClementPinard/SfmLearner-Pytorch)

---

## Thoughts

- It's really stupid to compute camera motion using a network: Camera motion estimation is a well established problem and there are many reliable methods to solve this problem.
- Also we need to come up with a method to aggregate information from multiple images -- naive agregation of single image based depth estimation seems silly: We suffer from information loss while the processed information is used.
- In traditional SfM approaches, we either match sparse features and treat non-static scene, non-rigid objects or non-lambertian surfaces as outliers OR we explicitly model the distribution of degenrate cases (so the total distribution is binomial+Gaussian). However, such distribution can be very complicated and should be learned from data instead of manually modeled as a simple distribution. That being said, both the parameters of the distribution/likelihood $\theta$ AND the optimization variables, e.g., depth $D$ and pose  $g$ are unknowns.
- We can solve this optimization problem in an alternative fashion: During training, we are solving $\theta$ of the likelihood function. During inference, we can fix $\theta$ and maximize the likelihood (minimize the negative log-likelihood) by solving $D, g$.
