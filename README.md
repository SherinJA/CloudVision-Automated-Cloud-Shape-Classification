# CloudVision: Automated Cloud Shape Classification


## Project Overview

Accurate cloud classification is essential for effective weather prediction, traditionally requiring manual analysis by meteorologists. **CloudVision** is an automated system designed to classify cloud shapes using a Convolutional Neural Network (CNN) with advanced image preprocessing techniques. By leveraging deep learning, the system aims to reduce the workload of meteorologists and enhance the efficiency and accuracy of cloud classification for weather forecasting.

**Authors**: Mengyang Zhao, Chorng Hwa Chang, Wenbin Xie, Zhou Xie, Jinyong Hu  
**Students**: Ananyaa Kyra Srikanth (21pt01), Shwetha S (21pt24), Sherin J A (21pt28)

## Objectives

The primary goal of CloudVision is to classify static cloud observation photos into three main cloud types:
- Cumulus
- Cirrus
- Stratus

This automation streamlines meteorological analysis, improving the speed and reliability of cloud-based weather predictions.

## Methodology

CloudVision employs a single-channel CNN with robust preprocessing and training strategies to achieve accurate cloud classification:

- **Preprocessing**:
  - **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Enhances image contrast to emphasize cloud features.
  - **SkyAware Transform**: A custom transformation that darkens non-sky regions based on blue channel intensity, focusing on cloud areas.
  - **Data Augmentation**: Includes random horizontal/vertical flips, rotations (up to 45°), color jitter, random resized crops (256x256), Gaussian blur, and random erasing to improve model generalization.

- **CNN Classifier**:
  - Utilizes **EfficientNet-B0**, a pre-trained CNN model, with frozen early layers to leverage transfer learning.
  - Features a custom classifier with dropout (0.5) and a linear layer outputting three classes (Cumulus, Cirrus, Stratus).
  - Incorporates weight decay for regularization.

- **Training Optimizations**:
  - Employs the **AdamW** optimizer with gradient clipping to ensure stable training.
  - Uses a **ReduceLROnPlateau** scheduler to adjust the learning rate when validation accuracy plateaus.
  - Implements **early stopping** to prevent overfitting by halting training if validation performance stagnates.

## Results

The system was trained and validated on a static dataset of cloud images, demonstrating promising classification accuracy. The preprocessing techniques (CLAHE and SkyAware Transform) effectively enhanced cloud features, contributing to reliable performance. The use of transfer learning with EfficientNet-B0 and training optimizations ensured efficient convergence and robust generalization.

## Dataset

The dataset used for training and validation is the **Cirrus Cumulus Stratus Nimbus (CCSN) Database** available on Kaggle:  
[https://www.kaggle.com/datasets/mmichelli/cirrus-cumulus-stratus-nimbus-ccsn-database](https://www.kaggle.com/datasets/mmichelli/cirrus-cumulus-stratus-nimbus-ccsn-database)

- **Class: Cirrus** - 894 images
- **Class: Stratus** - 885 images
- **Class: Cumulus** - 764 images

The dataset consists of static images, which are preprocessed and split into training (80%) and validation (20%) sets.

## Installation

To run the CloudVision system, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/cloudvision.git
   cd cloudvision
   ```

2. **Install dependencies**:
   Ensure you have Python 3.8+ installed. Install the required packages using:
   ```bash
   pip install torch torchvision numpy matplotlib opencv-python pillow
   ```

3. **Download the dataset**:
   - Download the CCSN dataset from the Kaggle link above.
   - Place the dataset in the `./cloud_dataset/cloud_dataset_3` directory or update the `root_dir` path in the code.

## Usage

1. **Prepare the dataset**:
   Ensure the dataset is organized in the expected directory structure for the `CloudDataset` class (update the code if necessary).

2. **Run the training script**:
   Execute the main script to train the model:
   ```bash
   python cloudvision.py
   ```
   The script will:
   - Load and preprocess the dataset.
   - Train the EfficientNet-B0 model for up to 20 epochs.
   - Save the best model weights to `best_cloud_model.pth`.
   - Display plots of training loss and accuracy.

3. **Evaluate the model**:
   The script automatically evaluates the model on the validation set during training. To test on new images, modify the code to include an inference pipeline.



## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

Please ensure your code follows the project’s style guidelines and includes
