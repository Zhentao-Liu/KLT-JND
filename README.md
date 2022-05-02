# Towards Top-Down Just Noticeable Difference Estimation of Natural Images
This is a brief introduction of our proposed JND profile with totally new design philosophy based on KLT. You can change our program as you like and use it anywhere, but please refer to its original source and cite our paper.

# Table of content
1. [Link](#Link)
2. [Abstract](#Abstract)
3. [Download](#Download)
4. [Requirement](#Requirement)
5. [Questions](#Questions)
6. [Citation](#Citation)

# Link
- Title: Towards Top-Down Just Noticeable Difference Estimation of Natural Images 
- Publish: IEEE Transactions on Image Processing, 2022
- Authors: Qiuping Jiang, Zhentao Liu, Shiqi Wang, Feng Shao, Weisi Lin
- Institution: The School of Information Science and Engineering, Ningbo University
- Paper: [2022-TIP-KLTJND](https://github.com/Zhentao-Liu/KLT-JND/raw/main/2022-TIP-KLTJND.pdf)

# Abstract
Just noticeable difference (JND) of natural images refers to the maximum pixel intensity change magnitude that typical human visual system (HVS) cannot perceive. Existing efforts on JND estimation mainly dedicate to modeling the diverse masking effects in either/both spatial or/and frequency domains, and then fusing them into an overall JND estimate. In this work, we turn to a dramatically different way to address this problem with a top-down design philosophy. Instead of explicitly formulating and fusing different masking effects in a bottom-up way, the proposed JND estimation model dedicates to first predicting a critical perceptual lossless (CPL) counterpart of the original image and then calculating the difference map between the original image and the predicted CPL image as the JND map. We conduct subjective experiments to determine the critical points of 500 images and find that the distribution of cumulative normalized KLT coefficient energy values over all 500 images at these critical points can be well characterized by a Weibull distribution. Given a testing image, its corresponding critical point is determined by a simple weighted average scheme where the weights are determined by a fitted Weibull distribution function. The performance of the proposed JND model is evaluated explicitly with direct JND prediction and implicitly with three applications including JND-guided noise injection, JND-guided image compression, and distortion detection and discrimination. Experimental results have demonstrated that promising performance of the proposed JND model.

# Download
You can download our proposed software in this project page directly.

# Requirement
Matlab

# Questions
If you have any problem of our program, please feel free to contact with the authors: jiangqiuping@nbu.edu.cn, zhentaoliu0319@163.com.

# Citation
If you find this work is useful for you, please cite our paper. Bibtex type citation will be available soon.
