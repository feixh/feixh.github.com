---
layout: site
title: Keypoint Detection
permlink: /notes/
exclude: true
---

### Single Image 3D Interpreter Network

Jiajun Wu, ECCV 16.

- 3 stage training:
    - 2d keypoint detection with keypoint annotation on real images as supervision
    - skeleton model parameter regression with (rendered keypoints, parametrized skeleton model) pairs as supervision
    - end-to-end training of the complete pipeline with projection layer at the end of the network
- Datasets: 
    - Pascal3D, Yu Xiang
    - FLIC for human bodies: Multimodal decomposable models for human pose estimation
    - CUB-200-2011 for birds: The Caltech-UCSD Birds-200-2011 Dataset
    - Keypoint-5: own dataset
- Keypoint detection:
    - FLIC: Percentage of Correct Keypoints (PCK), [35,45,44] [44] has toolkits for evaluation
    - CUB-200-2011: Percentage of Correct Parts (PCP) and Average Error (AE), [26, 38], [26] provides the evaluation code
- Structural parameter estimation:
    - Quantitative: 
        - [61] convex relaxation, Xiaowei Zhou
        - [25] IKEA dataset
    - Qualitative:
        - Keypoint-5, IKEA, SUN

---

### 3D Shape Estimation from 2D Landmarks: A Convex Relaxation Approach

Xiaowei Zhou, CVPR 15.

- Nonrigid structure from motion (NRSfM) via matrix factorization + metric rectification [9, 39]; recent iterative methods [29, 15, 2]
- Model: 
    - Weak perspective projection: simply ignore depth component (or assume depth variation over the object surface is negligible compared to the distance between the object and the camera)
    - Shape as a linear combination of basis shapes, each of which is composed of a fixed number of (predefined) landmarks in a canonical reference frame. One needs to also estimate the rigid body transformation to algin the canonical reference frame to the camera frame, in addition to the coefficients of the linear combination.
    - The basis shapes form an over-complete dictionary of shapes (in contrast to earlier works where only a small set of shapes is considered). To ensure only a small set of shapes is selected from the over-complete dictionary, l1 norm on the coefficients is imposed as regularization.
- Input is 2d landmarks with known 3d association, i.e., the data association problem is not solved in this paper.

---

### Learning Dense Correspondence via 3D-guided Cycle Consistency

---

### Unsupervised learning of object landmarks by factorized spatial embeddings

<http://openaccess.thecvf.com/content_ICCV_2017/papers/Thewlis_Unsupervised_Learning_of_ICCV_2017_paper.pdf>


James Thewlis, ICCV 17.

- TPS (Thin plane spline) is used in [30] WarpNet: weakly supervised matching for single-view reconstruction. 
- Relation to existing landmark detection algorithm: this work can build on top of existing detectors and can be used as a pre-training strategy to learn landmarks with no or less supervision. -- How to understand this statement? 
- Equivariance constraint: [37] Learning covariant feature detectors. TODO: read this one.

Method is very interesting. Let $S \subset \mathbb{R}^3$ be the surface of an object which is potentially subject to deformation and rigid pose transformation. Let $\Phi: \mathbb{R}^3 \mapsto \mathbb{R}^2$ be a mapping from the surface to image domain. Essentially, $\Phi$ should be shape-specific, thus we associate it with $S$ as subscript, i.e., $\Phi_S$. However, since only the image of the object $I: \mathbb{R}^2 \mapsto \mathbb{R}_+$ but not the underlying shape is known, $\Phi_S$ is parameterized by the image, i.e., $\Phi_S(\cdot; I)$. Now assume the image $I'$ differs $I$ by a domain transformation/warp $W: \mathbb{R}^2\mapsto \mathbb{R}^2$, for the same surface point $X\in S$, we have its coordinates in image $I$: $x=\Phi_S(X; I)$ and in image $I'$: $x'=\Phi_S(X; I'=I\circ W)$ and the two image coordinates are related by $x'=W(x)$. Therefore, we have the following equivaraince constraint:

$$
\Phi_S(X; I) = W\big( \Phi_S(X; I\circ W) \big).
$$

Image pairs as long as the domain warp are needed in training. They construct the training set by warping each image via a randomly generated transformation $W$, thus they have both the image pair and warp.

- The equivariance constraint is a generalization of the relation between images of points on static rigid bodies. It is related to the reprojection error in unsupervised learning of depth/flow/disparity prediction.
    - TODO: write down the explicit relation.

- In the paper, it is claimed that the learned landmarks are consistent across different instances of the same category and qualitative results are present. However, from the figure, it seems even though the intra-category shape variability is large, the viewpoint change is very small. 
    - SHOULD PLAY WITH THEIR MODEL ON OBJECTS AT POSES DIFFERENT FROM THOSE SHOWN IN THE PAPER.
- The method is also compared to supervised landmark detectors. To do that, an extra regressor is trained to map the detected landmarks (unsupervised learned, may not have semantic meanings) to ground truth landmarks annotated in the dataset. Detection accuracy/error is comparable to state-of-the-art supervised methods, but does NOT outperform strong baselines.

- Come back for the effective implementation of loss.
- Code is not available (?) -- could play with this with tensorflow.

- TODO: read those which cite this one.

---

### Unsupervised Discovery of Object Landmarks as Structural Representations

<https://arxiv.org/pdf/1804.04412.pdf>

Yuting Zhang, CVPR 18.

- Drawback of Thewlis is: 
    - Their method did not explicitly encourage the landmarks to appear at critical locations for image modeling.
    - Non-differentiable? (go back to Thewlis)
- TODO: Look at WarpNet, which is mentioned by both this and Thewlis
- In addition to keypoints, feature maps are also extracted. Both the geometric (keypoints) and photometric (feature maps) are used to reconstruct the image, which is involved in the reconstruction error. This is meaningful for applications like image reconstruction, but not really useful for pose estimation, where geometry is the only concern, NOT appearance.
- Other losses include the equivariance loss and separance loss, which are also used by Thewlis, and concentration loss which penalizes spatial variance of each landmark.
- Check out the arXiv version of the paper (48 pages) which contains a lot of details on training and qualitative results in the appendix.

---

### Unsupervised learning of object frames by dense equivariant image labelling

<http://papers.nips.cc/paper/6686-unsupervised-learning-of-object-frames-by-dense-equivariant-image-labelling.pdf>

James Thewlis, NIPS 17.

- Related work: [44] Deep Deformation Network: improves over WarpNet by using a *Point Transformer Network*
- Dense embedding rather than discrete one (Thewlis CVPR 17)

---

### Statistical transformer networks: Learning shape and appearance models via self supervision

<https://arxiv.org/pdf/1804.02541.pdf>

Anil Bas, arXiv

---

### WarpNet: weakly supervised matching for single-view reconstruction

---

### Learning covariant feature detectors

---

### PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes






