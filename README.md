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

segment_featurizer.py reads from a folder of images (each with a corresponding mask in a nested folder), and generates a .csv file containing the full feature set and percent-segment-masked for each segment_number-beta value pair. The .csv files are saved in the features/segment_features folder.

train_segment_regressors.py trains an XGBoost Regressor for each .csv file (corresponding to each n-beta value pair) and pickles the models into the models/segment_regressors folder.

Once trained, a given segment regressor can be applied to a non-masked segment. The output of the regression describes how much an input segment resembles previousley seen masked segments in the segment-feature space defined in feature_schema.txt: higher values mean that a segment looks more like masked, or lesion-containing segments. If the regressor is applied to every segment in a segmented image, the regressor output for each segment can be used to generate a heatmap. 

picture of heatmaps for different value pairs

#### Classify Heatmaps

heatmap_featurizer.py has the same structure as segment_featurizer.py, except that it converts a heatmap into a set of features (a different set from the segment features! see feature_schema.txt). For training images, the label is simply the given category - much simpler than the segment regression.  It saves the heatmap features for a given segmentation parameter value set (segment_number-beta pair) as a .csv in the features/heatmap_features folder.

train_heatmap_classifiers.py trains an XGBoost Classifer for each .csv file created by heatmap_featurizer.py. These classifiers are pickled in the models/heatmap_classifiers folder.

Once a full set of classifiers has been trained, they can be applied to the output of the segment regressors for a given test image. To determine the final classification for an image we simply take an up-down majority vote and assign the image to the winning class.

#### Localization

Finally, if an image is classified as containing a lesion, I attempt to localize it by applying a bounding box. At present this is done by simply thresholding the averaged heatmap and identifying the largest contour remaining. This approach will fail when there are multiple lesions in a single image or when heat from a lesion is below the threshold (usually because it is more distributed). 

Currently this step is more a visualization perk, however it could be developed to add a further level of machine-assistance to the user, for example by indicating in which direction to move the ultrasound probe in order to better characterize the l
esion. 

## Setup

I used Anaconda 4.3, python 2.7, and OpenCV 2.4.11. OpenCV is mainly used for visualization and can be removed entirely from the model if necessary. 

These scripts expect a file structure as found in this repo. The models and features folders currently contain a subset of the models and features that I used - a full set can be shared upon request (paulgamble09@gmail.com). 

To train your own models, you'll need ultrasound images hand-sorted into normal and lesion categories (I used ~12,000). You'll also need hand-drawn masks for some number of lesion images (I used ~500). In order to protect patient privacy, iSono has asked that I not share the original dataset. All images are used with permission.

### Using Current (Pretrained) Model

Edit apply_full_pipeline.py to point to your data. This will look for a full complement of models in the corresponding folders (segments and heatmaps). This will output the final classification. 

apply_localizer.py works exactly the same way except that it also returns an image of the averaged heatmap and the input image, with a bounding box added if the image was classified as likely to contain a lesion. 

### Train Your Own

Place your images and masks into the corresponding folders. Then run segment_featurizer.py and heatmap_featurizer.py. [Note: heatmap_featurizer.py took roughly an hour to run on my laptop]. Once all of the feature .csv are in place, you can train the corresponding XGBoost Regressors and Classifiers with train_segment_regressors.py and train_heatmap_classifiers.py

With these models in place, apply_full_pipeline.py and apply_localizer.py should work as above. 



