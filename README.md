# CMSC-733
Projects for the course CMSC733 - Classical and Deep Learning Approaches for
Geometric Computer Vision

## HW0: Alohamora!
This was an introductory project for the course and was divided into two phases:

#### Phase 1: Shake My Boundary
Boundary detection is an important, well-studied computer vision problem. Clearly it would be nice to have algorithms which know where one object transitions to another. But boundary detection from a single image is fundamentally diffcult. Determining boundaries could require object-specific reasoning, arguably making the task hard. A simple method to find boundaries is to look for intensity discontinuities in the image, also known of edges.

In this homework, a simplified version of [probability of boundary detection algorithm](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/papers/amfm_pami2010.pdf) was developed, which finds boundaries by examining brightness, color, and texture information across multiple scales (different sizes of objects/image). The output of the algorithm is a per-pixel probability of the boundary detected. The simplified algorithm performs much better when compared to classical edge detection algorithms like [Canny](https://ieeexplore.ieee.org/document/4767851) and [Sobel](https://en.wikipedia.org/wiki/Sobel_operator). 
<!--![original](Abhi1625_hw0/Phase1/BSDS500/Images/8.jpg)![pblite](Abhi1625_hw0/Phase1/Code/8/PbLite_8canny=0.1.png) -->

<img src="Abhi1625_hw0/Phase1/BSDS500/Images/8.jpg" align="center" alt="Your image title" width="400"/> <img src="Abhi1625_hw0/Phase1/Code/8/PbLite_8canny=0.1.png" align="center" alt="Your image title" width="400"/>

## HW1: AutoCalib

## P1 : My AutoPano

## P2 : FaceSwap
