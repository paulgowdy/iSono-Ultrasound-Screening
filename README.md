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

#### Segment Image

First, an input image is segmented using spectral_clustering (sklearn.cluster). A user-determined family of clustering parameters is applied to each image. Currently I only vary the number of segments and beta (roughly the bending energy of the segment boundaries), but other parameters could be used. I chose the following sets of values:

n_segments = [35, 40, 45, 50, 55, 60]</br>
betas = [5, 6, 7, 8, 9, 10]

This produces a family of segmented images (see multi_segment in is_utils.py).
<h3 align="center">
<img src="https://github.com/paulgowdy/Insight-BoS17/blob/master/segment_example.png" width="400">
</h3>
make this a family!

From this point forward, segment_number-beta value combinations are held apart: the steps below are repeated seperately for each of the 36 possible n-beta pairs.

#### Segment Regression

Segments are individually featurized. A full list of segment features can be seen in feature_schema.txt located in the features folder. By overlaying the corresponding lesion mask onto a segmented image, we can calculate the fraction of each segment covered by the mask. This fraction describes what percent of the segment consists of lesion-like tissue and will serve as our label when we train a segment-regressor. 

picture of mask overlay (faded)

segment_featurizer.py points to a folder of images (each with a corresponding mask in a nested folder), and generates a .csv file containing the full feature set and percent-segment-masked for each segment_number-beta value pair. The .csv files are saved in the features/segment_features folder.

train_segment_regressors.py trains an XGBoost Regressor for each .csv file (corresponding to each n-beta value pair) and pickles the models into the models/segment_regressors folder.

Once trained, a given segment regressor can be applied to a non-masked segment. The output of the regression describes how much an input segment resembles previousley seen masked segments in the segment-feature space defined in feature_schema.txt: higher values mean that a segment looks more like masked, or lesion-containing segments. If the regressor is applied to every segment in a segmented image, the regressor output for each segment can be used to generate a heatmap. 

picture of heatmaps for different value pairs

#### Classify Heatmaps

heatmap_featurizer.py

train_heatmap_classifiers.py

### Using Current (Pretrained) Model

### Train Your Own


## Setup

I used Anaconda 4.3, python 2.7, and OpenCV 2.4.11. OpenCV is mainly used for visualization and can be removed entirely from the model if necessary. 

These scripts expect a file structure as found in this repo. The models and features folders currently contain a subset of the models and features that I used - a full set can be shared upon request (paulgamble09@gmail.com). 

To train your own models, you'll need ultrasound images hand-sorted into normal and lesion categories (I used ~12,000). You'll also need hand-drawn masks for some number of lesion images (I used ~500). In order to protect patient privacy, iSono has asked that I not share the original dataset. All images are used with permission.


