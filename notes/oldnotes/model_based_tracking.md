---
layout: site
title: Model-based Tracking 
permlink: /notes/
exclude: true
---

# Edge-based Method

## Pose Optimization in Edge Distance Field for Textureless 3D Object Tracking

- An OK paper, in Computer Graphics International (not a top conference?), can be searved as an index to related work
- Related work:
  1. Pixel-wise tracker PWP3D[18]: PWP3D: Real-Time Segmentation and Tracking of 3D Objects
  2. state-of-the-art tracker GOS[24]: Global Optimal Searching for Textureless 3D Object Tracking.
  3. Utilize GPU[11]: Full-3D Edge Tracking with a Particle Filter.
  4. Most related[3]: 3D Textureless Object Detection and Tracking: An Edge-based Approach.
  5. Early work, since which people started using edge-based method to track objects[9]: CONDENSATION - Conditional Density Propagation for Visual Tracking. IJCV 1998.
  6. Keypoint for initialization and edge for alignment refinement[4]: Robust 3D Visual Tracking using Particle Filtering on the Special Euclidean Group: A Combined Approach of Keypoint and Edge Features.
  7. RAPid - A Video-Rate Object Tracker. BMVC 1990.
  8. Distance transform[6]: Distance Transforms of Sampled Functions.

- Two previous papers mentioned:
  1. **Optimal Local Searching for Fast and Robust Textureless 3D Object Tracking in Highly Cluttered Backgrounds**
  2. **Global optimal searching for textureless 3D object tracking**


## Full-3D Edge Tracking with a Particle Filter

- A good entry point for edge-based GPU-accelerated, real-time tracking
- Problem of edges as features to track: one image edge looks much like another.
- CONDENSATION algorithm with annealing [12]: Real-time camera tracking using known 3D models and a particle filter. ICPR 2006.

## Real-Time Camera Tracking Using Known 3D Models and a Particle Filter

- Edge matching is performed by 1-dimensional searches along normals to the projected lines of the model. The optimization robustly minimizes an accumulation of the disparities along the normals.
- Particle annealing: We begin with a tolerant observation density and iteratively increase its discrimination. We simultaneously adjust each particle's camera pose with a decreasing amount of randomness. This causes particles to group around strong modes in the more discriminating likelihood. In our context, we begin with large thresholds on the distance and irection of candidate edge point matches. We then reduce these thresholds at each iteration of anealing.
- Hidden line removal is not applied in thiw work, thus they can only handle simple objects like boxes.

## [Augmenting markerless complex 3D objects by combining geometrical and color edge information](http://www.irisa.fr/lagadic/pdf/2013_ismar_petit.pdf)

## 3D Textureless Object Detection and Tracking: An Edge-based Approach

- Most closely related work. Key reference.

## BB8: A Scalable, Accurate, Robust to Partial Occlusion Method for Predicting the 3D Poses of Challenging Objects without Using Depth

- Dataset entry:
  - LINEMOD (the PAMI paper, same as LINE-2D): Gradient Response Maps for RealTime Detection of Textureless Objects. PAMI 2012.
  - T-LESS (industry textureless objects): T-LESS: An RGB-D Dataset for 6D Pose
Estimation of Texture-less Objects. WACV 2017.
  - Occlusion: Learning 6D Object Pose Estimation Using 3D Object Coordinates. ECCV 2014.

- Key related work:
  - Idea inspired by [3]
  - Pose refinement[17]
  - State-of-the-art on LINEMOD[2]
  - Occlusion dataset[1]
  - SOA on T-LESS using color and depth[10]


# Region-based Method
---


## PWP3d, Bibby and Reid
- Most recent works are based on these two papers.
- First one proposed a GPU based real time version for pixelwise posterios 3d tracking and 2d segmentation, which is baed on the theoretical foundations discovered in the latter.

## 2D-3D Pose Estimation of Heterogeneous Objects Using a Region Based Approach

Jonathan Hexner, Rami R, Hagege, IJCV 2016

- Use local color histograms instead of global color histogram to model images


## 3d object tracking via boundary constrained region-based model

ICIP 2014

- Useful references:
  - dataset: Blort-the blocks world robotic vision toolbox, ICRA 2010 [code](https://github.com/pal-robotics/perception_blort)
  - comparison: Real-time modelbased rigid object pose estimation and tracking combining dense and sparse visual cues, CVPR 2013 [dataset](http://www.karlpauwels.com/datasets/rigid-pose/)

- Consider coherence of nearby pixels. Not much different from PWP3D.

## Model Based Training, Detection and Pose Estimation of Texture-Less 3D Objects in Heavily Cluttered Scenes

ACCV 10

- Came up with the ACCV LineMod dataset
- The procedure to generate templates seems useful for initialization procedure.
- Directly related: Multimodal Templates for Real-Time Detection of Texture-less Objects in Heavily Cluttered Scenes

## Real-Time 3D Tracking and Reconstruction on Mobile Phones

Trans. on Visualization and Computer Graphics, 2015.

- PWP3d on mobile phones
- Joint tracking and reconstruction of unknown objects.
- FIXME: read and fill in details.

## A Geometric Approach to Joint 2D Region-Based Segmentation and 3D Pose Estimation Using a 3D Shape Prior

- Level set based method, before PWP3d, might have some insights
- But state of the  art is based on PWP3d


## Datasets
- http://cvlab-dresden.de/research/scene-understanding/pose-estimation/#ACCV14
- http://www.karlpauwels.com/
