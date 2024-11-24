# Cross-Modal Space Learning with MLP Aggregation for Image-Text Matching 🔗🖼️📝

![Idea](Images/idea.png)

## 🌟 Abstract
In the realm of artificial intelligence, image-text matching remains a pivotal yet challenging task. Traditional methods often struggle to align visual and textual data effectively. This paper introduces **Cross-Modal Space Learning with MLP Aggregation**, a novel approach that embeds images and texts into a unified representation space, enabling the capture of subtle relationships between the two modalities. Key contributions include:  
- **Cross-modal learning** for bridging the semantic gap.
- **MLP-based aggregation** for efficient and precise matching.
- Superior performance on benchmark datasets, significantly improving recall rates.

This research advances multimodal AI by providing a robust framework for accurate and efficient image-text matching.

---

## 🏗️ Architecture Overview
![Architecture](Images/main.png)

Our proposed model follows these core steps:
1. **Image and Text Embedding**: Map visual and textual data into a unified latent space.
2. **Cross-Modal Aggregation**: Use an MLP-based architecture to bridge the semantic gap between image and text representations.
3. **Recall Optimization**: Leverage an objective function to optimize the matching task, ensuring high recall and efficiency.

---

# 📂 Repository Contents
This repository contains the implementation of our approach, including datasets, pre-processing, training scripts, and evaluation methods.

### 🔥 **Firefly Algorithm Folder**: `firefly`
- **`fa_mincon.m`**: MATLAB implementation of the Firefly Algorithm, inspired by Xin-She Yang's *Nature-Inspired Metaheuristic Algorithms* (2010). It is used to optimize weights, as described in the paper section **Role of the Optimizer**.
- **`run_fa_mincon.m`**: Script to execute the Firefly Algorithm, optimizing the cost function defined in the project. It is used for training the model by finding the optimal weights.

### 📚 **Word Embedding**
- **`Word_Embeding.ipynb`**: 
  - This script processes image captions to extract visual words and their synonyms using **ConceptNet** and **SpaCy**. 
  - The output is saved in a `.pkl` file, which is later used in the **text fragments** section to improve processing in subsequent steps.

### 📷 **Image Tensor Generation**
- **`Tensor_of_image.ipynb`**: 
  - Loads images from the COCO dataset and uses Faster R-CNN to extract image tensors. 
  - This step is essential for generating image fragments and for obtaining the bounding boxes of detected objects.
  
- **`Object_Detection_using_Faster_RCNN_PyTorch.ipynb`**: Demonstrates how to select and visualize tensor images generated in the previous step. This file allows for the visualization of detected objects and their associated image tensors.

### 📊 **Evaluation and Metrics**
- **`Recall@K`**: This script tests the image and text tensors, calculating **Recall@K** to evaluate the performance of the model. The metric is explained in the paper section **Recall Metric** and is the primary evaluation measure for the approach.

### 🧰 **File Conversion**
- **`Open mat files.ipynb`**: Converts trained weights from MATLAB `.mat` files to Python-compatible `.npy` files. This conversion ensures compatibility between MATLAB (used in the Firefly algorithm) and Python (used for tasks like Recall evaluation and MLP training).

### 🧠 **Model Training and Testing**
- **`MLP.ipynb`**: Implements the MLP (Multilayer Perceptron) architecture for aggregation. This file is used to:
  - **Train** the model.
  - **Evaluate** the model's performance after training.
  
- **`CodeForAll.ipynb`**: Combines all the components from the previous files into one unified script to test the final results. It provides an easy way to run the entire pipeline and evaluate its performance.

---

## 🚀 Getting Started
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/cross-modal-space-learning
   cd cross-modal-space-learning

   ```
2. **Install Dependencies**:
This project requires several Python packages and MATLAB dependencies. You can install the necessary Python libraries by using pip:

```bash
pip install -r requirements.txt
```
For MATLAB, ensure that you have the required toolbox for running the Firefly algorithm and any other necessary functions in .m files.

Prepare the Datasets: Download the necessary datasets (e.g., COCO, ConceptNet) and place them in the designated directories as specified in the project. The Word_Embeding.ipynb script will use these datasets to extract visual words and their synonyms.

**Running the Code:**

Firefly Algorithm:
To run the Firefly optimization for training:
Run run_fa_mincon.m in MATLAB to start the optimization of weights using the Firefly Algorithm:
```bash
matlab -nodisplay -r "run('firefly/run_fa_mincon.m'); exit"
```
**Word Embedding**:
The Word_Embeding.ipynb script is used for processing image captions and saving the output in a .pkl file. Run the Jupyter Notebook in Python:
```bash
jupyter notebook Word_Embeding.ipynb
```
**Image Tensor Generation:**
To generate image tensors from the COCO dataset using Faster R-CNN:
Run Tensor_of_image.ipynb to extract tensors from images:

```bash
jupyter notebook Tensor_of_image.ipynb
```
Use Object_Detection_using_Faster_RCNN_PyTorch.ipynb to visualize the tensors and bounding boxes of detected objects.

**Model Training:**
To train the MLP model and evaluate it:

Run MLP.ipynb:

```bash
jupyter notebook MLP.ipynb

```
**File Conversion:**
To convert .mat files (from MATLAB) to .npy files (for use with Python):

Run Open mat files.ipynb:

```bash

jupyter notebook Open mat files.ipynb
```
**Full Pipeline:**
To run the complete pipeline, which combines all the steps:

Run CodeForAll.ipynb to execute the entire pipeline:

```bash

jupyter notebook CodeForAll.ipynb
```
## ✨ Key Results
- **High Recall@K:** Outperformed baseline methods on standard datasets.
- **Efficient Aggregation:** Reduced computational complexity without sacrificing accuracy.
