# PIQ23: An Image Quality Assessment Dataset for Portraits

![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2FDXOMARK-Research%2FPIQ2023&label=VISITORS&countColor=%23f47373&labelStyle=upper)

This is the official repo for PIQ23, accepted in CVPR2023.

<img src=Imgs/CVRP%20Logo_2023%20Vancouvar_Color.png width='350'> &emsp;&emsp; <img src=Imgs/youtube.avif width='120'> &emsp;&emsp; <img src=Imgs/poster.png width='105'><br/>
&ensp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&ensp;&ensp;[CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/html/Chahine_An_Image_Quality_Assessment_Dataset_for_Portraits_CVPR_2023_paper.html)&ensp;&ensp; &ensp;&emsp; &emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;&emsp;&ensp;&ensp;&ensp;&ensp;[Video](https://youtu.be/cvWjOWq5wnk)&ensp; &emsp;&emsp;&emsp;&emsp;&emsp; &emsp;[Poster](Imgs/CVPR_Poster_PIQ23.png)

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
 - The PIQ23 dataset (5GB) can be downloaded from the DXOMARK CORP [**website**](https://corp.dxomark.com/data-base-piq23/).
 - You need to fill the form and agree to the terms and conditions in order to request access to the dataset. We garantee open access to any individual or institution following these instructions.
 - In a short time, your request will be validated and you will receive an automatic email with a temporary link in order to download the dataset.

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
Each folder is associated to an attribute (Details, Overall and Exposure). It contains the images of the corresponding regions of interest with the following naming: {img_nb}\_{scene_name}\_{scene_idx}.{ext}. 

The CSV files include the following entries: 
- **IMAGE PATH**: relative path to the image ({Attribute}\\{Image name})
- **IMAGE**: image name
- **JOD**: jod score of the image
- **JOD STD**: jod standard deviation
- **CI LOW**: lower bound of image's confidence interval
- **CI HIGH**: upper bound of image's confidence interval
- **CI RANGE**: CI HIGH - CI LOW
- **QUALITY LEVEL**: preliminary quality level (result of the clustering over CIs)
- **CLUSTER**: final quality levels (result of the variance analysis and community detection)
- **TOTAL COMPARISONS**: total number of comparisons for this image
- **SCENE**: scene name
- **ATTRIBUTE**: attribute (Exposure, Details or Overall)
- **SCENE IDX**: scene index (from 0 to 49)
- **CONDITION**: lighting condition (Outdoor, Indoor, Lowlight or Night)

## Test Splits
We provide two **official** test splits for PIQ23:
- **Device split**:
    - We split PIQ23 by devices, in order to test the general performance of the trained models on the given scenes.
    - The test set contains around 30% of images from each scene, thus 30% of the whole dataset. 
    - To avoid device bias, we have carefully selected devices from different quality levels and price ranges for the test set. This split can still include some images of the test devices in the training set and vice versa since the distribution of devices per scenes is not completely uniform. We can garantee that more than 90% of the training and testing devices do not overlap.
    - We first sort the devices by their median percentage of images across scenes then split them into five groups from the most common device to the least and sample from these five groups until we get around 30% of the dataset.
    - The device split csv can be found in "Test split\Device Split.csv".
    - The test and train csv for the different attributes can be found here "Test split\Device Split\".
- **Scene split**: 
    - We split PIQ23 by scene in order to test the generalization power of the trained models.
    - We have carefully chosen 15/50 scenes for the testing set, covering around 30% of the images from each condition, thus 30% of the whole dataset, around 1486/5116 images.
    - To select the test set, we first sort the scenes by the percentage of images in the corresponding condition (Outdoor, Indoor, Lowlight, Night), we then select a group of scenes covering a variety of condition (framing, lighting, skin tones, etc.) until we get around 30% of images for each condition.
    - The scene split csv can be found in "Test split\Scene Split.csv".
    - The test and train csv for the different attributes can be found here "Test split\Scene Split\".
    - Examples of the test and train scenes can be found in "Test split\Scene Split\Scene examples".

An example of how to use the splits can be found in the "Test split example.ipynb" notebook. 

***NB:*** 
- Please ensure to publish results on both splits in your papers.
- The paper's main results cannot be reproduced with these splits. We will be publishing official performances on these splits soon.

## Benchmarks

**Note on the results**:
- The models chosen as *optimal* in this experience are the ones who scored a maximum SROCC on the testing sets. Please take into consideration that a maximum SROCC does not reflect a maximum in other parameters. 
- Other experiences where the optimal model is chosen based on a combination of metrics would be optimal.
- There should be a margin of error taken into account in these metrics. A difference of certain percentage in correlation can be due to multiple factors and might not be repeatable.

<table>
  <tr>
    <th colspan="13">Device Split</th>
  </tr>
  <tr>
    <th rowspan="2">Model\Attribute</th>
    <th colspan="4">Details</th>
    <th colspan="4">Exposure</th>
    <th colspan="4">Overall</th>
  </tr>
  <tr>
    <th>SROCC</th>
    <th>PLCC</th>
    <th>KROCC</th>
    <th>MAE</th>
    <th>SROCC</th>
    <th>PLCC</th>
    <th>KROCC</th>
    <th>MAE</th>
    <th>SROCC</th>
    <th>PLCC</th>
    <th>KROCC</th>
    <th>MAE</th>
  </tr>
  <tr>
    <td>DBCNN (1200 x LIVEC)</td>
    <td>0.787</td>
    <td>0.783</td>
    <td>0.59</td>
    <td>0.777</td>
    <td>0.807</td>
    <td>0.804</td>
    <td>0.611</td>
    <td>0.704</td>
    <td>0.83</td>
    <td>0.824</td>
    <td>0.653</td>
    <td>0.656</td>
  </tr>
  <tr>
    <td>MUSIQ (1200 x PAQ2PIQ)</td>
    <td>0.824</td>
    <td>0.831</td>
    <td>0.65</td>
    <td><b>0.627</b></td>
    <td><b>0.848</b></td>
    <td><b>0.859</b></td>
    <td><b>0.671</b></td>
    <td><b>0.585</b></td>
    <td><b>0.848</b></td>
    <td>0.837</td>
    <td>0.65</td>
    <td>0.626</td>
  </tr>
  <tr>
    <td>HyperIQA (1344 (224*6) x No IQA pretraining)</td>
    <td>0.793</td>
    <td>0.766</td>
    <td>0.618</td>
    <td>0.751</td>
    <td>0.8</td>
    <td>0.828</td>
    <td>0.636</td>
    <td>0.721</td>
    <td>0.818</td>
    <td>0.825</td>
    <td>0.66</td>
    <td><b>0.612</b></td>
  </tr>
  <tr>
    <td>SEM-HyperIQA (1344 (224*6) x No IQA pretraining)</td>
    <td><b>0.854</b></td>
    <td><b>0.847</b></td>
    <td><b>0.676</b></td>
    <td>0.645</td>
    <td>tbd</td>
    <td>tbd</td>
    <td>tbd</td>
    <td>tbd</td>
    <td>0.845</td>
    <td><b>0.856</b></td>
    <td><b>0.674</b></td>
    <td>0.641</td>
  </tr>
</table>

<table>
  <tr>
    <th colspan="13">Scene Split</th>
  </tr>
  <tr>
    <th rowspan="2">Model\Attribute</th>
    <th colspan="4">Details</th>
    <th colspan="4">Exposure</th>
    <th colspan="4">Overall</th>
  </tr>
  <tr>
    <th>SROCC</th>
    <th>PLCC</th>
    <th>KROCC</th>
    <th>MAE</th>
    <th>SROCC</th>
    <th>PLCC</th>
    <th>KROCC</th>
    <th>MAE</th>
    <th>SROCC</th>
    <th>PLCC</th>
    <th>KROCC</th>
    <th>MAE</th>
  </tr>
  <tr>
    <td>DBCNN (1200 x LIVEC)</td>
    <td>0.623</td>
    <td>0.55</td>
    <td>0.449</td>
    <td><b>0.916</b></td>
    <td>0.712</td>
    <td>0.741</td>
    <td>0.528</td>
    <td><b>0.863</b></td>
    <td>0.633</td>
    <td>0.621</td>
    <td>0.446</td>
    <td>1.173</td>
  </tr>
  <tr>
    <td>MUSIQ (1200 x PAQ2PIQ)</td>
    <td>0.674</td>
    <td><b>0.671</b></td>
    <td>0.487</td>
    <td>0.972</td>
    <td><b>0.764</b></td>
    <td><b>0.772</b></td>
    <td><b>0.569</b></td>
    <td>0.87</td>
    <td>0.736</td>
    <td><b>0.759</b></td>
    <td>0.549</td>
    <td><b>0.956</b></td>
  </tr>
  <tr>
    <td>HyperIQA (1344 (224*6) x No IQA pretraining)</td>
    <td><b>0.701</b></td>
    <td>0.668</td>
    <td><b>0.504</b></td>
    <td>0.936</td>
    <td>0.692</td>
    <td>0.684</td>
    <td>0.498</td>
    <td><b>0.863</b></td>
    <td><b>0.74</b></td>
    <td>0.736</td>
    <td><b>0.55</b></td>
    <td>0.989</td>
  </tr>
</table>


## Citation
Please cite the paper/dataset as follows:
```bibtex
@InProceedings{Chahine_2023_CVPR,
    author    = {Chahine, Nicolas and Calarasanu, Stefania and Garcia-Civiero, Davide and Cayla, Th\'eo and Ferradans, Sira and Ponce, Jean},
    title     = {An Image Quality Assessment Dataset for Portraits},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {9968-9978}
}

```
## License
Provided that the user complies with the Terms of Use, the provider grants a limited, non-exclusive, personal, non-transferable, non-sublicensable, and revocable license to access, download and use the Database for internal and research purposes only, during the specified term. The User is required to comply with the Provider's reasonable instructions, as well as all applicable statutes, laws, and regulations.

## About
For any questions please contact: piq2023@dxomark.com


