# Code of project "Semantic Segmentation on WildScene: UNet, SegNet, DeepLabV3+, and SVD Tuning"

## Installation

We use the package manager [pip](https://pip.pypa.io/en/stable/) to initialize the environment.

```bash
pip install numpy
pip install opencv-python
pip install opencv-contrib-python
pip install matplotlib
pip install scikit-learn
pip install torch
pip install torchvision
pip install segmentation-models-pytorch
pip install pytorch-lightning
```

## Location of the Dataset

We place all the data in WildScenes dataset under the `dataset` directory. In structure:
```
├── dataset
│   ├── image
│   ├── indexLabel
│   ├── label
```
Due to the file size limitation, we didn't include this folder and it will need to be recreated.

Also, we have a toy dataset with only 5 images for `Unet_SVD_toy_dataset.ipynb`, in folders `image_small` and `indexLabel_small`.

## .py and .ipynb File
We have 8 code files containing the code used for different steps and models.

* `split.py`
  * The code for splitting the dataset into some smaller datasets.

* `Analysis.py`
  * The code for frequency research.

* `Preprocessing.ipynb`
  * The code and demo images of `White Balance` and `Flares, Shadow and Lighting conditions` section in the Preprocessing part.

* `UNet_ACWE.py`
  * The code of UNet with ACWE boundaries.

* `Unet_SVD_toy_dataset.ipynb`
  * Training and testing code with outputs and demo images of UNet and UNet with SVD. (Using toy dataset)

* `UNet.ipynb`
  * Training and testing code with outputs and demo images of UNet and UNet with SVD.

* `SegNet.ipynb`
  * Training and testing code with outputs and demo images of SegNet.

* `DeepLabV3+.ipynb`
  * Training and testing code with outputs and demo images of DeepLabV3+.

## Trained Weights File
UNet: https://unsw.sharepoint.com/:u:/s/comp9517279/ESlC9LLecmlFtbqX85rfVZMBXUPe0g3VHQmmeUvCblhaQQ?e=LdQm9n

UNet with SVD tuning: https://unsw.sharepoint.com/:u:/s/comp9517279/ETBUSHI0RFJFlxgzXti1_ugBS9-4w2zdHEaVrxthWMEDtg?e=XvrfZe

SegNet: https://unsw.sharepoint.com/:u:/s/comp9517279/EbI1ydrlJthMljCzw5uHZYgBMv1QrK-ijke9ERjzuBrAlg?e=vGq7KA

DeepLabV3+: 

## Note
* We have confirmed that all codes can work. If not, please contact us.