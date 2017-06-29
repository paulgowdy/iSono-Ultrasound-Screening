# Machine Assisted Breast Cancer Screening
Insight Data Science consulting project, Summer 2017

Develped in collaboration with iSono Health, this image classification system detects suspicious breast lesions in ultrasound images. This repository contains a fully trained pipeline as well as files to train a new model on your own data. 

<h1 align="center">
<img src="https://github.com/paulgowdy/Insight-BoS17/blob/master/c16_bb.gif">
</h1>

![alt text](https://github.com/paulgowdy/Insight-BoS17/blob/master/c16_bb.gif)

## Setup

I used Anaconda 4.3, python 2.7, and OpenCV 2.4.11. OpenCV is mainly used for visualization and can be removed entirely from the model if necessary. 

These scripts expect a file structure as found in this repo. The models and features folders currently contain a subset of the models and features that I used - a full set can be shared upon request (paulgamble09@gmail.com). 

To train your own models, you'll need ultrasound images hand-sorted into normal and lesion categories (I used ~12,000). You'll also need hand-drawn masks for some number of lesion images (I used ~500). In order to protect patient privacy, iSono has asked that I not share the original dataset. All images are used with permission.

<center>![alt text](https://github.com/paulgowdy/Insight-BoS17/blob/master/c31_bb.gif)</center>
