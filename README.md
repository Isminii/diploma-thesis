# Brain Tumor Segmentation - Diploma Thesis Project

This project is part of my diploma thesis at the Department of Electrical and Computer Engineering, University of Patras. It presents **a deep learning-based approach for the automatic detection and segmentation of brain tumors into clinically significant subregions using 3D MRI scans**.
The goal is to support therapeutic planning, enhance treatment precision, and demonstrate the transformative potential of AI in healthcare applications.



## Installation

Follow these steps to set up and run the project locally:

### Prerequisites
Make sure you have **Python 3.x** and **pip** installed. It's recommended to create a **virtual environment** to avoid dependency conflicts.

### Steps to Install:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Isminii/diploma-thesis.git
   cd brain-tumor-segmentation

2. **Create a virtual enviroment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt

4. **Download the dataset:**

   This project uses the [BraTS 2021 Task 1 dataset](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1) available on Kaggle. It contains 3D 
   multimodal MRI brain scans with pixel-level tumor annotations for segmentation tasks.

   Each patient scan includes the following modalities:
    - **FLAIR**
    - **T1**
    - **T1c** (contrast-enhanced)
    - **T2**
    
    And a corresponding **segmentation mask** indicating tumor regions.

    You need to have a [Kaggle account](https://www.kaggle.com) and accept the dataset's terms of use to download it.



## Models

This project includes the implementation and evaluation of three deep learning models for brain tumor segmentation:

1. **3D U-Net**
   
   A volumetric version of the classic U-Net architecture, adapted to process full 3D MRI volumes. It captures spatial features across all three dimensions, making 
   it well-suited for medical image segmentation tasks.

3. **3D TransUNet**
   
   A hybrid architecture that combines a 3D U-Net with a Vision Transformer (ViT). It leverages the strengths of CNNs for local feature extraction and Transformers    for capturing long-range dependencies within 3D volumes.

5. **2D Slice-by-Slice TransUNet**
   
   A 2D variant of the TransUNet model that processes each MRI volume slice-by-slice instead of as a full 3D volume. It applies a 2D U-Net + ViT hybrid 
   architecture to individual slices and then reconstructs the predicted 3D segmentation mask by stacking the outputs from each slice.



## Project Structure 

The project is organized into separate folders for each model architecture, making it easy to manage code and experiments independently. Each **model folder** typically contains:

- `brats_dataloader.py`:
  Handles data loading, preprocessing, and augmentation for the BraTS dataset.

- `<model_name>.py`:
  Defines the architecture of the model.

- `<model_name>_training.py`:
  Contains the training loop, loss functions, optimizer setup, and logging.

- `<model_name>_testing.py`:
  Implements the evaluation pipeline and inference logic for the trained model.


## Training & Evaluation

To train a specific model, run the corresponding training script:
   
   ```python <model_name>_training.py```

To evaluate a trained model, use the associated testing script:

   ```python <model_name>_testing.py```








