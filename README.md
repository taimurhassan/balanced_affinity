# Balanced Affinity Loss For Highly Imbalanced Baggage Threat Instance Segmentation

## Introduction
This repository contains the implementation of our paper titled 'Balanced Affinity Loss For Highly Imbalanced Baggage Threat Instance Segmentation'. 

![CBA](/images/image1.png)

## Installation
To run the code, please download and install Anaconda (also install MATLAB R2020a with deep learning, image processing and computer vision toolboxes). Afterward, please import the ‘environment.yml’ or alternatively install following packages: 
1. Python 3.7.4 
2. TensorFlow 2.2.0 (CUDA compatible GPU needed for GPU training) 
3. Keras 2.3.1 or above 
4. OpenCV 4.2 
5. Imgaug 0.2.9 or above 
6. Tqdm 

Both Linux and Windows OS are supported.

## Datasets
The X-ray datasets can be downloaded from the following URLs: 
1. [GDXray](https://domingomery.ing.puc.cl/material/gdxray/) 
2. [SIXray](https://github.com/MeioJane/SIXray) 
3. [OPIXray](https://github.com/OPIXray-author/OPIXray) 
4. [COMPASS-XP](https://figshare.com/articles/dataset/COMPASS-XP/9249791)

Each dataset contains the ground truths either in mat files, txt files or in xml files. The proposed framework requires annotations to be in the mask form. Therefore, to parse each dataset annotations, we have provided their respective parser in the ‘…\Segmentation\utils’ folder. Please follow the same steps as mentioned below to prepare the training and testing data. These steps are also applicable for any custom dataset.

## Dataset Preparation

1. Download the desired dataset and update the dataset paths in ‘…\Segmentation\mst.m’ file.
2. Run the ‘…\Segmentation\mst.m’ file to generate the tensor representation of the input scans (this step is required for both train and test scans). 
3. Put the training tensors (obtained through step 2) in the '…\Segmentation\trainingDataset\train_images' folder. 
4. Put their corresponding training annotations in '…\Segmentation\trainingDataset\train_annotations' folder (Note: Use parsers provided in the 'utils' folder to obtain the annotations). 
5. Put validation images in '…\Segmentation\trainingDataset\val_images' folder 
6. Put validation annotations in '…\Segmentation\trainingDataset\val_annotations' folder 
7. Please note here that these images and annotations should have same name and extension (preferably png). 
8. Put test images in '…\Segmentation\testingDataset\test_images' folder and their annotations in '…\Segmentation\testingDataset\test_annotations' folder. 
9. The test images should also be obtained from the step 2 whereas the folder '…\Segmentation\testingDataset\original' should contain the respective original images (the final detection results are overlaid on these images). 
10. Dataset directory structure is given below:
```
├── Segmentation\trainingDataset
│   ├── train_images
│   │   └── tr_image_1.png
│   │   └── tr_image_2.png
│   │   ...
│   │   └── tr_image_n.png
│   ├── train_annotations
│   │   └── tr_image_1.png
│   │   └── tr_image_2.png
│   │   ...
│   │   └── tr_image_n.png
│   ├── val_images
│   │   └── va_image_1.png
│   │   └── va_image_2.png
│   │   ...
│   │   └── va_image_m.png
│   ├── val_annotations
│   │   └── va_image_1.png
│   │   └── va_image_2.png
│   │   ...
│   │   └── va_image_m.png
├── Segmentation\testingDataset
│   ├── original
│   │   └── o_image_1.png
│   │   └── o_image_2.png
│   │   ...
│   │   └── o_image_k.png
│   ├── test_images
│   │   └── te_image_1.png
│   │   └── te_image_2.png
│   │   ...
│   │   └── te_image_k.png
│   ├── test_annotations
│   │   └── te_image_1.png
│   │   └── te_image_2.png
│   │   ...
│   │   └── te_image_k.png
│   ├── segmentation_results
│   │   └── te_image_1.png
│   │   └── te_image_2.png
│   │   ...
│   │   └── te_image_k.png
```

## Training and Testing
1. Use '…\Segmentation\trainer.py' file to train the backbone network provided in the '…\Segmentation\codebase\models' folder. The training parameters can be configured in this file as well. Once the training is completed, the segmentation results are saved in the '…\Segmentation\testingDataset\segmentation_results' folder. These results are used by the 'Segmentation\instanceDetector.m' script in the next step for bounding box and mask generation. 
2. Once the step 1 is completed, please run '…\Segmentation\instanceDetector.m' to generate the final detection outputs. Please note that the '…\Segmentation\instanceDetector.m' requires that the original images are placed in the '…\Segmentation\testingDataset\original' folder (as discussed in step 12 of the previous section).

## Results


![feature](/images/latent1.png)

<h3 align="center"> Feature Distribution of the Proposed Balanced Affinity Loss Function </h3>

![qual](/images/Vis2.png)

<h3 align="center"> Qualitative Examples of the Proposed System Across Different Datasets </h3>

Please feel free to email us if you require the trained instances. 

## Citation
If you use the proposed system (or any part of this code in your research), please cite the following paper:

```
@inproceedings{cba,
  title   = {Balanced Affinity Loss For Highly Imbalanced Baggage Threat Instance Segmentation},
  author  = {Abdelfatah Ahmed and Ahmad Obeid and Divya Velayudhan and Taimur Hassan and Ernesto Damiani and Naoufel Werghi},
  note = {Under Review in IEEE ICASSP},
  year = {2022}
}
```

## Contact
If you have any query, please feel free to contact us at: taimur.hassan@ku.ac.ae (Taimur Hassan), 100058254@ku.ac.ae (Divya Velayudhan) or 100059689@ku.ac.ae (Abdelfatah Ahmed).
