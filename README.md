# Towards Top-Down Just Noticeable Difference Estimation of Natural Images
This is a brief introduction of our proposed JND profile with totally new design philosophy based on KLT. You can change our program as you like and use it anywhere, but please refer to its original source and cite our paper.

# Table of content
1. [Paper Link](#Paper-Link)
2. [Abstract](#Abstract)
3. [Download](#Download)
4. [Requirement](#Requirement)
5. [Questions](#Questions)
6. [Citation](#Citation)

# Paper Link
- Title: Towards Top-Down Just Noticeable Difference Estimation of Natural Images 
- Publish: IEEE Transactions on Image Processing, 2022
- Authors: Qiuping Jiang, Zhentao Liu, Shiqi Wang, Feng Shao, Weisi Lin
- Institution: The School of Information Science and Engineering, Ningbo University
- Link: Our paper will be accepted soon. To be continue.

# Abstract
Just noticeable difference (JND) of natural images refers to the maximum change magnitude that the typical human visual system (HVS) cannot perceive. Existing efforts on JND estimation mainly dedicate to modeling the visibility masking effects of different factors in either/both spatial or/and frequency domains, and then fusing them into an overall JND estimate. In this work, we turn to a dramatically different way to address these problems with a top-down design philosophy. Instead of formulating and fusing multiple masking effects in a bottom-up way, the proposed JND estimation model dedicates to first predicting a critical perceptual lossless (CPL) counterpart of the original image and then calculating the difference map between the original image and the predicted CPL image as the JND map. We conduct subjective experiments to determine the critical points of 500 images and find that the distribution of cumulative normalized KLT coefficient energy values over all 500 images
at these critical points can be well characterized by a Weibull distribution. Given a testing image, its corresponding critical point is determined by a simple weighted average scheme where the weights are determined by a fitted Weibull distribution function. The performance of the proposed JND model is evaluated explicitly with direct JND prediction and implicitly with two applications including JND-guided noise injection and JND-guided image compression. Experimental results have demonstrated that our proposed JND model can achieve better performance than several latest JND models.

# Download
You can download our proposed software in this project page in the near future. Once our papar is accepted, we will upload it immediately.

# Requirement
Matlab

# Questions
If you have any problem of our program, please feel free to contact with the authors: jiangqiuping@nbu.edu.cn, zhentaoliu0319@163.com.

# Citation
If you find this work is useful for you, please cite our paper. Bibtex type citation will be available soon.
