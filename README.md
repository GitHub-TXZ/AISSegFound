# AISSegFound

## A Customized Foundation Model for Acute Ischemic Stroke Lesion Segmentation on Multimodal CT Scans

This repository provides the implementation and experimental workflows for the paper:

> **AISSegFound: Customized Foundation Model with Mixture-of-Experts Fine-Tuning and Missing-Modality Adaptation for Acute Ischemic Stroke Lesion Segmentation on Multimodal CT Scans**

AISSegFound is a customized foundation-model framework for acute ischemic stroke lesion segmentation on multimodal CT scans. Built upon [SegVoL]([BAAI-DCAI/SegVol: The official code for "SegVol: Universal and Interactive Volumetric Medical Image Segmentation".](https://github.com/BAAI-DCAI/SegVol)), it integrates Mixture-of-Experts (MoE) fine-tuning and missing-modality adaptation to enhance multimodal feature learning and achieve robust segmentation under both complete and incomplete modality settings.

---

## Contents

- [Installation](#1. Installation)
- [Pretrained Weights](#2. Pretrained Weights)
- [Dataset Preparation](#3. Dataset Preparation)
- [Preprocessing and Dataset Planning](#4. Preprocessing and Dataset Planning)
- [Training and Validation](#5. Training and Validation)
- [Inference](#6. Inference)
- [Acknowledgements](#Acknowledgements)

---

## 1. Installation

### 1.1 Clone the repository

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/GitHub-TXZ/AISSegFound.git
cd AISSegFound
```

### 1.2 Create a virtual environment

Python 3.9 is recommended for this project to ensure compatibility with the MONAI version used by AISSegFound. Using Python 3.9 helps avoid potential dependency and compatibility issues.

```bash
conda create -n aissegfound python=3.9
conda activate aissegfound
```

### 1.3 Install PyTorch and CUDA Dependencies

The reference environment is based on PyTorch 2.1.2 and CUDA 11.8. Install the corresponding PyTorch packages before installing the remaining project dependencies.

```bash
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```

The installed PyTorch and CUDA runtime should be compatible with the available NVIDIA GPU driver. The CUDA runtime version bundled with PyTorch does not necessarily need to match the locally installed CUDA Toolkit version exactly.

### 1.4 Install Project Dependencies

Install the Python dependencies specified by the repository:

```bash
pip install -r requirements.txt
```

Ensure that the installation completes successfully before proceeding to dataset preparation and model training.

## 2. Pretrained Weights

AISSegFound uses pretrained SegVoL weights to initialize the foundation model. Before training, download the appropriate pretrained checkpoint and place it in the following directory:

```
AISSegFound/
└── network/
    └── pretrained_weights/
        └── ViT_pretrain.ckpt
```

**Important**: The pretrained SegVol checkpoint is required to initialize the foundation model. Please refer to the [official SegVol repository]([BAAI-DCAI/SegVol: The official code for "SegVol: Universal and Interactive Volumetric Medical Image Segmentation".](https://github.com/BAAI-DCAI/SegVol)) for instructions on obtaining the pretrained weights. After downloading the checkpoint, place it in the `network/pretrained_weights` directory and ensure that the checkpoint path matches the configuration used by AISSegFound.

## 3. Dataset Preparation

AISSegFound adopts the dataset organization convention used by nnU-Net v2. Each dataset is assigned a unique numeric identifier and a descriptive name, and each imaging modality is stored as a separate channel file.

### 3.1 Dataset Naming Convention

The dataset directory follows the naming pattern:

```
Dataset<ID>_<Name>
```

Replace `<ID>` with the dataset identifier and `<Name>` with a descriptive dataset name.

For example:

```
Dataset001_Test
```

The raw dataset should be organized under the `nnUNet_raw` directory. For example:

```
nnUNet_raw/
└── Dataset001_Test/
    ├── imagesTr/
    ├── labelsTr/
    ├── imagesTs/
    └── dataset.json
```

Here:

- `imagesTr/` contains the training images.
- `labelsTr/` contains the corresponding ground-truth segmentation masks.
- `imagesTs/` optionally contains images for inference or testing. (Optional)
- `dataset.json` specifies the dataset metadata, modality names, label definitions, and image file format.

The dataset identifier used during preprocessing, training, validation, and inference must be consistent.

### 3.2 Image Organization

Each case must have a unique identifier shared by all its modalities and its corresponding segmentation label. Following the nnU-Net convention, modality files are distinguished by four-digit channel suffixes, such as `_0000`, `_0001`, and `_0002`.

```
nnUNet_raw/
└── Dataset001_Test/
    ├── imagesTr/
    │   ├── case_001_0000.nii.gz
    │   ├── case_001_0001.nii.gz
    │   ├── case_001_...
    │   ├── case_002_0000.nii.gz
    │   ├── case_002_0001.nii.gz
    │   └── ...
    ├── labelsTr/
    │   ├── case_001.nii.gz
    │   ├── case_002.nii.gz
    │   └── ...
    ├── imagesTs/  
    │   ├── case_101_0000.nii.gz
    │   ├── case_101_0001.nii.gz
    │   └── ...
    └── dataset.json
```

The channel suffixes must agree with the ordering defined in `dataset.json` and with the input-channel configuration expected by the model.

All images belonging to the same case should be spatially aligned, and each segmentation label must correspond to the same case. Please check image dimensions, voxel spacing, orientation, and spatial geometry to avoid inconsistencies during preprocessing and training.

### 3.3 Configure `dataset.json`

Create a `dataset.json` file in the dataset root directory. For the format of `dataset.json`, you can organize it according to the following example:

```json
{
    "labels": {
        "background": 0,
        "ischemic_stroke_lesion": 1
    },
    "name": "Test",
    "numTraining": 5,
    "channel_names": {
        "0": "CT_CBF",
        "1": "CT_CBV",
        "2": "CT_MTT",
        "3": "CT_Tmax",
        "4": "CT" 
    },
    "file_ending": ".nii.gz"
}
```

The main fields are described below:

- `channel_names`: Defines the imaging modalities and their channel indices. These indices must correspond to the file suffixes used in `imagesTr/` and `imagesTs/`.
- `labels`: Defines the segmentation classes and their integer values.
- `numTraining`: Specifies the number of training cases with available labels.
- `file_ending`: Specifies the file extension of the image and label volumes.
- `name`: Provides a descriptive name for the dataset.

Update `numTraining`, `name`, and the channel definitions to match your actual dataset. The example above uses five training cases only for illustration.

**Note:** If your customized nnU-Net implementation expects additional metadata or a different label schema, follow the requirements of the implementation used in this repository.

## 4. Preprocessing and Dataset Planning

AISSegFound uses the nnU-Net-style preprocessing and planning workflow to prepare the raw dataset for model training. This stage checks dataset consistency, analyzes the image properties, and generates the preprocessed data and configuration files needed by subsequent experiments.

### 4.1 Run Preprocessing

An example preprocessing command is:

```bash
CUDA_VISIBLE_DEVICES=0 \
python nnunetv2/experiment_planning/plan_and_preprocess_entrypoints.py \
    -d 001 \
    --verify_dataset_integrity
```

Here:

- `CUDA_VISIBLE_DEVICES=0` selects the GPU visible to the process, where GPU numbering follows the local environment.
- `-d 001` specifies the dataset identifier.
- `--verify_dataset_integrity` enables dataset integrity verification before preprocessing.

Replace `001` with the identifier of your actual dataset. The dataset identifier used here must match the directory under `nnUNet_raw`.

Depending on the repository configuration, the preprocessed dataset may be stored under `nnUNet_preprocessed/` or a project-specific data directory. Use the paths configured by the local implementation consistently throughout the workflow.

### 4.2 Prepare `splits_final.json`

The provided preprocessing workflow does not automatically generate the `splits_final.json` file required by the project's training setup. This file must be created manually and placed in the preprocessed dataset directory expected by the implementation.

For example:

```
data/
└── nnUNet_preprocessed/
    └── Dataset001_Test/
        └── splits_final.json
```

For the format of `splits_final.json`, you can organize it according to the following example:

```json
[
    {
        "train": [
            "case_1",
            "case_2",
            "case_3",
            "case_4"
        ],
        "val": [
            "case_5",
            "case_6"
        ]
    },
    ...
]
```

The case identifiers must match the case names recognized by the preprocessed dataset, without modality suffixes or file extensions.

Each split contains:

- `train`: The cases used to optimize the model parameters.
- `val`: The cases used for validation and model assessment.

The example illustrates the structure of a single split, rather than a recommended split ratio. For actual experiments, use an appropriate training-validation strategy based on the dataset size and study design.

When performing cross-validation, ensure that the split definitions are compatible with the requested fold indices and that no case appears in both the training and validation subsets of the same fold.

## 5. Training and Validation

AISSegFound follows the training workflow of nnU-Net v2 while incorporating project-specific model configurations and fine-tuning settings.

The training procedure uses a designated dataset, configuration, cross-validation fold, and experiment name. The experiment name can encode customized batch size, patch size, and spacing information, depending on the implementation.

### 5.1 Training Configuration

An example training command is shown below:

```bash
CUDA_VISIBLE_DEVICES=0 \
python nnunetv2/run/run_training.py \
    -dataset_name_or_id Dataset001_Test \
    -configuration 3d_fullres \
    -fold 0 \
    -ex_name Ex1@b_1_p_32_256_256_s_10.0_0.8600000143051147_0.8600000143051147
```

A second example uses the numeric dataset identifier and enables validation:

```bash
CUDA_VISIBLE_DEVICES=0 \
python nnunetv2/run/run_training.py \
    -dataset_name_or_id 001 \
    -configuration 3d_fullres \
    -fold 0 \
    -ex_name Ex1@b_1_p_32_256_256_s_10.0_0.8600000143051147_0.8600000143051147 \
    --val
```

Here:

- **-dataset_name_or_id** : Dataset identifier or full dataset name, such as `001` or `Dataset001_Test`.

- **-configuration** : Segmentation configuration, such as `3d_fullres` for full-resolution 3D segmentation.

- **-fold** : Cross-validation fold index, commonly `0`–`4` for five-fold cross-validation, or `all` if supported by the customized training script.

- **-ex_name** : specifies the experiment name and allows customized training settings to override the corresponding values in the experiment plans.

  For example, in `Ex1@b_1_p_32_256_256_s_10.0_0.8600000143051147_0.8600000143051147`:

  - `Ex1`: An experiment identifier used to distinguish different experiments.
  - `b_1`: Batch size of 1.
  - `p_32_256_256`: Patch size of (32 × 256 × 256).
  - `s_10.0_0.8600000143051147_0.8600000143051147`: Target spacing values used for preprocessing and training configuration.

- **--val** : Enables the validation behavior implemented by the customized training entry point.

- **--c** : Resumes training from a previously saved checkpoint, allowing the training process to continue rather than starting from scratch.

These commands illustrate the training entry point and the experiment configuration used by the repository. Confirm that the dataset identifiers, experiment names, and command-line arguments match the implementation in your checkout.

### 5.2 Multiple-fold training

For cross-validation experiments, the following shell loop illustrates how to run training across folds 0–4:

```bash
for f in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 \
    nnUNetv2_train \
        -dataset_name_or_id 001 \
        -configuration 3d_fullres \
        -fold "$f" \
        -ex_name Ex1@b_1_p_32_256_256_s_10.0_0.8600000143051147_0.8600000143051147
done
```

Each fold uses a different training-validation partition. Cross-validation can provide a more robust assessment of model performance when the dataset is limited, as is often the case in medical imaging research.

## 6. Inference

After training, AISSegFound can be used to generate segmentation predictions for unseen multimodal CT cases.

The inference input directory follows the same channel-naming convention as the training dataset. Each case must contain the input modalities expected by the trained model, and the modality suffixes must match the channel configuration used during training.

### 6.1 Prepare Inference Data

An example test directory is:

```
data/
└── nnUNet_raw/
    └── Dataset001_Test/
        └── imagesTs/
            ├── case_101_0000.nii.gz
            ├── case_101_0001.nii.gz
            ├── case_101_0002.nii.gz
            ├── case_101_0003.nii.gz
            ├── case_101_0004.nii.gz
            └── ...
```

Each case should have a unique identifier shared across its modalities. 

### 6.2 Run Inference

An example inference command is:

```bash
CUDA_VISIBLE_DEVICES=0 \
python nnunetv2/inference/predict_from_raw_data.py \
    -i data/nnUNet_raw/Dataset001_Test/imagesTs \
    -o data/test_output \
    -c 3d_fullres \
    -d 001 \
    -f 0 \
    -device cuda \
    -ex_name fold_0_Ex1@b_1_p_32_256_256_s_10.0_0.8600000143051147_0.8600000143051147
```

Here:

- **-d** : Dataset identifier or name associated with the trained model.
- **-i** : Input directory containing the multimodal CT images to be segmented.
- **-o** : Output directory for the predicted segmentation masks.
- **-c** : Network configuration, consistent with the training configuration.
- **-f** : Fold or folds used for inference, depending on the supported implementation.
- **-device** : Inference device, such as `cuda` or `cpu`, if supported.
- **-ex_name** : Experiment identifier used to locate the corresponding trained model and configuration.

Ensure that the experiment name and fold are consistent with the checkpoint you intend to use. The exact checkpoint directory and output format depend on the customized inference implementation.

The output directory will contain the segmentation predictions generated for the input cases. These predictions can subsequently be compared with reference annotations using appropriate segmentation evaluation metrics, such as the Dice similarity coefficient and lesion-wise metrics, where applicable.

### 6.3 Missing-Modality Configuration

AISSegFound incorporates missing-modality adaptation as part of its design. Please do not assume that an arbitrary subset of input channels can be omitted simply by removing the corresponding files. Follow the model's expected input format and the missing-modality handling logic implemented in the repository when preparing incomplete multimodal scans.

## Acknowledgements

We thank Fabian Isensee and the nnU-Net contributors for developing and maintaining the nnU-Net framework, and Yuxi Du for contributions to SegVoL. Their work provides important foundations for the development of AISSegFound. Please refer to the original projects and publications for their respective methods, implementations, and licensing terms.

