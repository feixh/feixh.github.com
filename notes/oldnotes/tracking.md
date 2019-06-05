---
layout: site
title: Tracking
permlink: /notes/
exclude: true
---

## End-to-end representation learning for Correlation Filter based tracking

- Correlation is an efficient algorithm that learns to discriminate an image patch from the surrounding patches by solving a large ridge regression problem extremely efficiently [4, 14]
- Proved to be highly successful in object tracking [6, 19, 23, 2], where efficiency of CF enables per frame update of the internal model
- It owes its speed to a Fourier domain formulation, which allows the ridge regression problem to be solved with a few applications of the FFT and cheap element-wise operations. -- much more efficient than iterative solver such as SGD, and still allows the discriminator to be tailored to a specific video, while embedding methods learns the representation offline.
- Combine CNN feature and CF [22, 7, 9, 32]: CNN and CF are complementary which leads to better performance when combined. Previous works don't have CNN-CF in an end-to-end system.
- Fully-convolutional Siamese network: feels similar to FCN semantic segmentation?
- General tracking procedure:
  1. Feed both the target (template) and search region (rectangle centered at previous tracked location and 4 times of the size of the previous tracked region) of the current frame to FC-Siamese, obtain a response map.
  2. Update tracked location with argmax of the response map.
  3. Use the updated tracked location as new template, proceed to next frame and start from step 1 again. (no update to the tracked region? only center?)
- Novelty of this paper is derive the differentials of the CF transformation such that CNN and CF can be trained end-to-end.
- For the derivation of differentials, see the paper.

---


## Fully-Convolutional Siamese Networks for Object Tracking

- Learning to track can be addressed using similarity learning, which learns a function $f(z, x)$ that compares an example image $z$ to a candidate image $x$ of the same size and returns a hight score if the two images depict the same object and a low score otherwise.
- Use CNN as the function $f$. Similarity learning with CNN is typically addressed using Siamese architecture, which applies the same transformation $\phi$ to both inputs and combine the representations using another function $g$, such that $f(z, x)=g(\phi(x), \phi(x))$. When $g$ is a simple distance or similarity metric, the function $\phi$ can be thought as an embedding.
- A function is *fully-convolutional* if it commutes with translation. Introduce $L_\tau$ to denote the translation operator $(L_\tau x)[u]=x[u-x]$, a function $h$ that maps signals to signals is fully-convolutional with integer stride $k$ if $h(L_{k\tau} x)=L_\tau h(x)$ for any translation $\tau$.
- Advantage of a fully-convolutional network: instaed of a candidate image of the same size, one can provide as input to the network an arbitrary size search image and compute the score map in a single evaluation. To achieve this, we use a convolutional embedding function $\phi$ and combine the resulting feature maps via a cross-correlation layer
$$
f(z, x) = \phi(z) * \phi(x) + b\mathbb{1}
$$
- Logistic loss is used: Given the predicted socre map $v:\Omega \mapsto \mathbb{R}$ and ground truth label map $y: \Omega \mapsto \{-1, +1\}$ the loss is
$$
L(y, v) = \frac{1}{|\Omega|}\sum_{u\in\Omega}\ell(y[u], v[u])
$$
where $\ell(y, v)=\log(1+\exp(-yv))$.

---

## Visual Object Tracking using Adaptive Correlation Filters

- To make the tracker fast, correlation is computed in the Fourier domain. First, apply 2D FFT to both input image the the filter (template). The Convolution Theorem states that correlation becomes element-wise multiplication in the Fourier domain. The bottleneck in this process is computing the forward and inverse FFTs so that the entire process has an upper bound time of $O(|\Omega|\log |\Omega|)$ where $\Omega$ is the tracking window.
- MOSSE Filter:
  1. To start, needs a set of training images $f_i$ and training outputs $g_i$. $g_i$ can have arbitrary shape in theory, here $g_i$ is a 2D Gaussian centered at the ground truth location of the tracking window.
  2. Training is conducted in the Fourier domain. $F_i, G_i$ denotes $f_i,g_i$ in Fourier domain and $H$ denotes the filter in Fourier domain.
$$
\begin{aligned}
G &= F\odot H^*\\
H_i^* &= \frac{G_i}{F_i}
\end{aligned}
$$
  To train a filter such that it maps input images to the desired ouputs, we solve the following minimization problem:
  $$
  \min_{H^*}\sum_i\| F_i\odot H^*-G_i\|^2
  $$
  To solve this minimization problem, one needs to take special care of the factor that all the variables are complex numbers. However, one can explicitly writes down the real and imaginary part of each variable and derive a closed-form solution.
  $$
  H^*=\frac{\sum_i G_i \odot F_i^*}{\sum_i F_i \odot F_i^*}
  $$
  For details, see the paper.
  <font color="yellow"> Normal equation for linear systems in complex domain? </font>
  3. To appreciate temporal consistency, add a momentum term:
  $$
  H_i^*=\eta \frac{G_i\odot F_i^*}{F_i \odot F_i^*}+(1-\eta)H_{i-1}^*
  $$

---

## High-Speed Tracking with Kernelized Correlation Filters
