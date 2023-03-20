# PIQ23: An Image quality Assessment Dataset for Portraits

## Introduction
We present PIQ23, a portrait-specific image quality assessment dataset of 5116 images of predefined scenes acquired by more than 100 smartphones, covering a high variety of brands, models, and use cases. The dataset features individuals from a wide range of ages, genders, and ethnicities who have given explicit and informed consent for their photographs to be used in public research. It is annotated by pairwise comparisons (PWC) collected from over 30 image quality experts for three image attributes: face detail preservation, face target exposure, and overall image quality.

## PIQ23

![thumb](Imgs/Thumbnail.png)

**Important Notes**
 - By downloading this dataset you agree to the terms and conditions.
 - All files in the PIQ23 dataset are available for non-commercial research purposes only.
 - You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the images and any portion of derived data.
 - You agree to remove, throughout the life cycle of the dataset, any set of images following the request of the authors.

 **Dataset Access**
 - The PIQ23 dataset can be download from the DXOMARK CORP [**website**](https://corp.dxomark.com/data-base-piq23/).
 - You need to fill the form and agree to the terms and conditions in order to request access to the dataset. We garantee open access to any individual or institution following these instructions.
 - In a short time, your request will be validated and you will receive an automatic email with a temporary link in order to download the dataset.
 - DXOMARK collects your information for **tracking purposes** only, in order to reach out to you in case some images need to be removed from the dataset.

 **Overview**

The dataset structure is as follows:  
```
├── Details 
├── Overall
├── Exposure
├── Scores_Details.csv
├── Scores_Overall.csv
└── Scores_Exposure.csv
```
Each folder is associated to an attribute (Details, Overall and Exposure). It contains the images of the corresponding regions of interest with the following naming: {img_nb}_{scene_name}_{scene_idx}.{ext}. 

The CSV files include the following entries: 
- **IMAGE PATH**: relative path to the image ({Attribute}\{Image name})
- **IMAGE**: image name
- **JOD**: jod score of the image
- **JOD STD**: jod standard deviation
- **CI LOW**: lower bound of image's confidence interval
- **CI HIGH**: upper bound of image's confidence interval
- **CI RANGE**: CI HIGH - CI LOW
- **QUALITY LEVEL**: preliminary quality level (result of the clustering over CIs)
- **TOTAL COMPARISONS**: total number of comparisons for this image
- **SCENE**: scene name
- **ATTRIBUTE**: attribute (Exposure, Details or Overall)
- **SCENE IDX**: scene index (from 0 to 49)
- **CONDITION**: lighting condition (Outdoor, Indoor, Lowlight or Night)

## Citation

## License

## About
For any questions please contact: nchahine@dxomark.com


