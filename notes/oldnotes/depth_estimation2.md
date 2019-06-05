---
layout: site
title: Depth Estimation 2
permlink: /notes/
exclude: true
---

## Unsupervised CNN for Single View Depth Estimation: Geometry to the Rescue

- Need pairs of images for training with small known relative motion. Train the convolutional encoder for the task of depth prediction for the source image. Reconstruct the source by warping the target image based on the predicted depth map and known motion. Photometric error in reconstruction is used as loss for the encoder.
- Model trained on less than half of the KITTI dataset gives comparable performance to the state-of-the-art.
- CNN for single-view depth estimation, first of its kind that can be trained end-to-end from scratch, in a fully *unsupervised* fashion, simply using data captured using a *stereo rig*.
- To handle the aperture problem: Adopt the simple $L_2$ regularization term to demonstrate the idea of unsupervised training with reconstruction loss, while various edge-preserving regularization terms can also be used.
- Coarse-to-fine training with *skip architecture* that combines the coarser depth prediction with the local information to get finer predictions.
- Error metric from Eigen's NIPS paper:
  - RMS: $\sqrt{\frac{1}{T}\sum_{i\in T}\| d_i-d_i^\text{gt}\|^2}$
  - log RMS: $\sqrt{\frac{1}{T}\sum_{i\in T}\| \log(d_i)-\log(d_i^\text{gt})\|^2}$
  - scale-invariant log RMS: In the equation above, replace $\log(d_i)$ with $(\log(d_i)-\bar\log(d_i))$ and $\log(d_i^\text{gt})$ with $(\log(d_i^\text{gt})-\bar\log(d_i^\text{gt}))$ respectively
  - Abs. Relative: $\frac{1}{T}\sum_{i\in T}\frac{|d_i-d_i^\text{gt}|}{d_i^\text{gt}}$
  - Sq. Relative: $\frac{1}{T}\sum_{i\in T}\frac{\|d_i-d_i^\text{gt}\|^2}{d_i^\text{gt}}$
  - Accuracies: \% of $d_i$ s.t. $\max(\frac{d_i}{d_i^\text{gt}},\frac{d_i^\text{gt}}{d_i})=\delta < \tau$
  - First three items: the smaller the better. Last item: the larger the better.
- Related:
  - [32] Deep3D: Fully automatic 2d-to-3d video conversion with deep convolutional neural networks.


---

## Depth Map Prediction from a Single Image using a Multi-Scale Deep Network

David Eigen, Christian Puhrsch, Rob Fergus

- NIPS 2014
- Employ two deep network stacks:
  - One makes a coarse global prediction based on the entire image
  - Another one refines the prediction locally
- Single-view depth estimation is not as well-established as stereo depth estimation:
  - The latter boils down to finding robust image point correspondences, which usually can be done via local appearance.
  - And the problem itself is ill-posed: Given an image, infinite number of possible scenes may produce it.
  - This work focuses on the *spatial relations* within a scene rather than general scale.
- Architecture and implementation details: come back later
  - Fully-connected layer after several convolutional layer in the coarse network.
  - Output of the coarse-scale network is concatenated to the first layer feature map of the fine-scale network, followed by several convolutional layers.
  - The final output of the fine-scale network has a resolution which is 1/4 of the input image.
- Scale-invariant mean squared error (in log space):
  - normalize log depth by taking off the mean
  - error is the sum of sequared error of normalized log depth

---

## DeMoN: Depth and Motion Network for Learning Monocular Stereo

- *Simple encoder-decoder network fails to make use of stereo: depth from a single image is a shortcut to satisfy the training objective.*

Does this explain the prefered structure used in the
<font color="red"> *left-right consistency* </font>
paper?
- Key to the shortcut problem is: An architecture that alternates optical flow estimation with the estimation of camera motion and depth.
- Related work section is very imformative
- Network architecture:
  - image pairs -> bootstrap bet -> iterative net -> refinement net -> final output (showed in fig.2)
  - structure of the bootstrap net \& iterative net (fig.3): two encoder-decoder units, first unit takes depth \& pose and outputs flow; second unit takes flow and spits out pose and depth; which are fed to the first unit again.
- Training schedule: see the paper for more details
- Comparison to classic structure from motion:
  - They implemented the pipeline themselves instead of using off-the-shelf ORB-SLAM, which makes the experiment results suspicious -- implementation of a state-of-the-art SfM pipeline is non-trivial
- Comparison to single-view depth estimation
  - DeMoN produces more detailed and regular depth images.
- Scene-specific priors, which are used by single-view based approach, might hurt depth estimation, while stereo based approach is independent of the content of the scene but only exploits the geometric relations between the image pairs, thus generalizes well to unknown scenes.


---

## DeepStereo: Learning to Predict New Views from the World's Imagery

- Generate pixel color of an unseen view by feeding neighboring views of a scene with associated poses.
  - In the sense of not explicitly producing depth image, DeepStereo is an implicit depth/optilcal flow estimation method.
- Minimal assumptions about the scene: the scene should be mostly static and should exist within a finite range of depths. Resulting images should degrade gracefully when assumptions are not satisfied.
- Approach:
  - Naively feeding the input images along with the camera poses to the network directly is unlikely to work well, since
    - The network needs to figure out how to interpret the pose parameters (rotation and translation) together with input pixel values, which is inefficient -- we know how to project pixels from multi-view geometry.
    - The network needs a lot of parameters to capture long-range connections between input pixels, which is inefficient. However, we can use the epiploar constraint to reduce such connections.
  - Leverage conventional plane-sweep approach:
    - Given a target camera pose, generate a plane-sweep volume for *each input view* at varying depth hypotheses. Each separate plane-sweep volume contains 4 channels: RGBA. Alpha is zero if the source pixel is unavailable.
    - So given $K$ input views, we have $K$ such volumes. And each volume is $M\times N\times D$ where $M,N$ is image size and $D$ is the depth level.
    - To decide the output pixel color at $(i,j)$, we need to consider the $(i,j)$ $D$-dim column of each plane-sweep volume.
    - Fig. 5, apply 2d convolution and ReLU over each depth layer of the input volumes in stage 1; stack the feature maps along the dimension of depth in stage 2 and apply convolution and ReLU.
- **Need to understand the network structure.**



---

## SfM-Net: Learning of Structure and Motion from Video






---

## Unsupervised monocular depth estimation with left-right consistency.

C. Godard, O. Mac Aodha, G.J.Brostow



---

## A large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation

---

## Predicting Depth, Surface Normals and Semantic Labels with a Common Multi-Scale Convolutional Architecture
