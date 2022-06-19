# COVID-19 COUGH SOUND CLASSIFICATION BASED ON DEEP LEARNING
===========================================================
This program is part of the final project of Zanjabila from Engineering Physics, Sepuluh Nopember Institute of Technology. The program here is a development of Edresson Casanova's program with a github link, namely github.com/Edresson/SPIRA-ComParE2021

This program uses 3 datasets, namely Coswara, Coughvid, ComParE CCS 2021. The whole program is made in python and runs on a jupyter notebook. Where there are several steps in running this overall program:
1. Run the 1_Download_Dataset.ipynb program to download the dataset you want to use.
2. Run the program 2_Create_CSV_Files.ipynb to prepare a csv datasheet for each dataset.
3. Run the 3_Normalized.ipynb program to normalize the audio in the dataset to eliminate variations that arise due to dataset differences.
4. Run the program 4_Cough_Detection.ipynb to detect cough sounds on the dataset.
5. Run the program 5_Cough_Detection.ipynb to detect and segment the cough sound.
6. Run the 6_Training_Testing.ipynb program to train and test the model. To perform experiments on the program, make changes to the desired directory path. If you only want to replicate the results, change the folder address to reload the experiment configuration that was done previously.

The following are the results that have been carried out in this program:

## Experimenting with Variations in Dataset
=====================================

ComParE + Coughvid + Coswara
- config path : json/variety-dataset/exp-1.json
- results : 59.46%

ComParE + Coughvid (positive) + Coswara
- config path : json/variety-dataset/exp-2.json
- result : 52.07%

ComParE + Coughvid + Coswara (positive)
- config path : json/variety-dataset/exp-3.json
- result : 63.21%

ComParE + Coughvid (positive) + Coswara (positive)
- config path : json/variety-dataset/exp-4.json
- results : 68.54%

## Experiments on Variations in Dataset Distribution (Train-Devel)
=====================================

70:30
- config path : json/variety-split/exp-1.json
- result : 65.18%

75:25
- config path : json/variety-split/exp-2.json
- result : 67.65%

80:20
- config path : json/variety-split/exp-3.json
- results : 68.14%

85:15
- config path : json/variety-split/exp-4json
- result : 72.68%

90:10
- config path : json/variety-split/exp-5.json
- results : 68.54%

95:5
- config path : json/variety-split/exp-6.json
- results : 69.14%

## Experiments Against Cough Detection Threshold
=====================================

60%
- config path : json/threshold-detection/exp-1.json
- result : 71.40%

70%
- config path : json/threshold-detection/exp-2.json
- results : 71.99%

80%
- config path : json/threshold-detection/exp-3.json
- result : 72.48%

90%
- config path : json/threshold-detection/exp-4.json
- result : 75.54%

## Experimenting the Segmentation Method
=====================================

Hysteresis Comparator
- config path : json/segmentation/exp-1.json
- result : 83.19%

Hysteresis Comparator Normalization
- config path : json/segmentation/exp-2.json
- result : 83.19%

RMS Threshold
- config path : json/segmentation/exp-3.json
- result : 83.19%

RMS Threshold Normalization
- config path : json/segmentation/exp-4.json
- result : 83.19%

## Experimenting the Augmentation Method
=====================================

No Noise Augmentation
- config path : json/augmentation/exp-1.json
- result : 86.36%

No Spec Augmentation
- config path : json/augmentation/exp-2.json
- result : 81.40%

Without both Augmentation
- config path : json/augmentation/exp-3.json
- result : 861.68%

## Experiment With Hyper-Parameter Tuning
=====================================

### Alpha Tuning
------------------------------------------------

0.1
- config path : json/alpha/exp-1.json
- result : 86.50%

0.2
- config path : json/alpha/exp-2.json
- result : 86.91%

0.3
- config path : json/alpha/exp-3.json
- results : 87.74%

0.4
- config path : json/alpha/exp-4.json
- result : 85.12%

0.5
- config path : json/alpha/exp-5.json
- result : 88.15%

0.6
- config path : json/alpha/exp-6.json
- results : 84.84%

0.7
- config path : json/alpha/exp-7.json
- result : 84.98%

0.8
- config path : json/alpha/exp-8.json
- result : 86.36%

0.9
- config path : json/alpha/exp-9.json
- result : 86.36%

1.0
- config path : json/alpha/exp-10.json
- result : 85.81%

### Tuning Learning Rate
------------------------------------------------

0.1
- config path : json/learning-rate/exp-1.json
- result : 53.53%

0.01
- config path : json/learning-rate/exp-2.json
- result : 49.17%

0.001
- config path : json/learning-rate/exp-3.json
- result : 88.15%

0.0001
- config path : json/learning-rate/exp-4.json
- result : 85.81%

0.00001
- config path : json/learning-rate/exp-5.json
- result : 49.41%

### Tuning Weight Decay
------------------------------------------------

0.1
- config path : json/weight-decay/exp-1.json
- result : 87.87%

0.01
- config path : json/weight-decay/exp-2.json
- result : 88.18%

0.001
- config path : json/learning-rate/exp-3.json
- result : 84.43%

0.0001
- config path : json/learning-rate/exp-4.json
- result : 86.36%

0.00001
- config path : json/learning-rate/exp-5.json
- result: 87,46
