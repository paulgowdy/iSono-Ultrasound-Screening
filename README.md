# Machine Assisted Breast Cancer Screening
Insight Data Science consulting project, Summer 2017

Develped in collaboration with iSono Health, this image classification system detects suspicious breast lesions in ultrasound images. This repository contains a fully trained pipeline as well as files to train a new model on your own data. 

<h1 align="center">
<img src="https://github.com/paulgowdy/Insight-BoS17/blob/master/c16_bb.gif" width="800">
</h1>

<h1 align="center">
<img src="https://github.com/paulgowdy/Insight-BoS17/blob/master/c31_bb.gif" width="800">
</h1>

## Model Description

First, an input image is segmented using spectral_clustering (sklearn.cluster). A user-determined family of clustering parameters is applied to each image. Currently I only vary the number of segments and beta (roughly the bending energy of the segment boundaries), but other parameters could be used. I chose the following sets of values:

n_segments = [35, 40, 45, 50, 55, 60]</br>
betas = [5, 6, 7, 8, 9, 10]

This produces a family of segmented images (see multi_segment in is_utils.py). An individual segmented image looks like this:

From this point forward, n_segment-beta value combinations are held apart: the steps below are repeated seperately for each of the 36 possible n-beta pairs.

Segments are individually featurized. A full list of segment features can be seen in feature_schema.txt located in the features folder. In order to train a 


### Using Current (Pretrained) Model

### Train Your Own


## Setup

I used Anaconda 4.3, python 2.7, and OpenCV 2.4.11. OpenCV is mainly used for visualization and can be removed entirely from the model if necessary. 

These scripts expect a file structure as found in this repo. The models and features folders currently contain a subset of the models and features that I used - a full set can be shared upon request (paulgamble09@gmail.com). 

To train your own models, you'll need ultrasound images hand-sorted into normal and lesion categories (I used ~12,000). You'll also need hand-drawn masks for some number of lesion images (I used ~500). In order to protect patient privacy, iSono has asked that I not share the original dataset. All images are used with permission.


