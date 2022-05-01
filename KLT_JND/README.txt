KLT Based JND Profile Software release.

========================================================================

-----------COPYRIGHT NOTICE STARTS WITH THIS LINE------------

Copyright (c) 2022 Ningbo University (NBU)
All rights reserved.

Permission is hereby granted, without written agreement and without license or royalty fees, to use, copy, 
modify, and distribute this dataset and code for any purpose, provided that the copyright notice in its 
entirety appear in all copies of this database and code, the research is to be cited in the bibliography as:

1)

IN NO EVENT SHALL NBU BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR 
CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF THIS DATASET AND CODE, EVEN IF NBU 
HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

NBU SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE DATASET AND 
CODE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, NBU HAS NO OBLIGATION TO PROVIDE 
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

-----------COPYRIGHT NOTICE ENDS WITH THIS LINE------------%

========================================================================

Author  : Zhentao Liu et al.
Version : 1.1

This folder includes the  implemantation of KLT based JND profile that we proposed. The algorithm is described in:

"Towards Top-Down Just Noticeable Difference Estimation of Natural Images"

You can change this program as you like and use it anywhere, but please refer to its original source  and cite our paper.

Document Description and Usage

Images :
The test images names as "Building.png". 

M file :
1.image_reshape.m
Reshape the image patches into a whole image.

2.img_scaled.m
Scale the JND map for finner observation.

3.KLT_JND.m
Generate the JND map of the original image.

4.modcrop.m
This is used to crop image border.

5.patch_extract.m
This is used for image patch extraction.

6.weibull_com.m
Compute the weibull distribution probability density function coefficient.

7.rum_me.m
Example program.

This code has been tested on Windows 10 in MATLAB R2019a 

========================================================================
