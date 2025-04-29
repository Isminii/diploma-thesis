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
   git clone https://github.com/your-username/brain-tumor-segmentation.git
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
   
