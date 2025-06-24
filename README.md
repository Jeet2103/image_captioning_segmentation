# 🖼️ Image Captioning and Segmentation App

An advanced and integrated deep learning application that can automatically **generate descriptive captions** for images and perform **instance segmentation** and **semantic segmentation** using state-of-the-art models like **BLIP** and **Mask R-CNN**. This project combines the power of computer vision and natural language processing to deliver an intuitive and interactive AI tool.

---

## 🌟 Project Overview

This project aims to bridge the gap between visual content and natural language by generating meaningful captions for images and identifying object segments. Built using powerful transformer-based models and CNN-based segmentation, the app empowers accessibility, searchability, and understanding of image content.

Users can upload an image and instantly receive:
- A **caption** describing the image context
- A **segmented version** highlighting different objects in the scene

All this is wrapped in a **Streamlit interface** for easy interaction.

---

## 🧠 Approach

1. **Dataset Preprocessing**
   - Used the **Flickr8k dataset** with captions stored in a CSV file.
   - Images are preprocessed with resizing, random flips, and color jitter for better generalization.
   - Captions are cleaned and tokenized with `BlipProcessor`.

2. **Model Training**
   - Captioning is powered by **BLIP (Bootstrapped Language Image Pretraining)**.
   - The BLIP model is fine-tuned using the processed Flickr8k dataset.
   - Segmentation uses **Mask R-CNN** (pretrained on COCO) for detecting and masking objects.

3. **Evaluation & Logging**
   - Captioning loss is tracked per epoch and visualized in loss graphs.
   - Segmentation and captioning losses are saved in `captioning/` and `segmentation/` directories.

4. **Streamlit App Integration**
   - Built an intuitive interface for users to upload images.
   - The app displays both the generated caption and segmented image.

5. **Modular Codebase**
   - Separated components for training, inference, and app interface.
   - Clean use of configuration files and utils to improve maintainability.

---

## 🛠️ Tech Stack

| Component         | Tool / Library                            |
|------------------|--------------------------------------------|
| Language          | Python 3.10                                |
| Deep Learning     | PyTorch, torchvision, transformers         |
| Data Handling     | pandas, datasets, PIL                      |
| Image Processing  | torchvision.transforms                     |
| Captioning Model  | [BLIP](https://github.com/salesforce/BLIP) |
| Segmentation Model| Mask R-CNN (ResNet50 FPN)                  |
| App Interface     | Streamlit                                  |
| Visualization     | Matplotlib                                 |
| Deployment        | Conda / Virtualenv                         |

---

## 🗂️ Codebase Structure

```bash
image-captioning-app/
├── config.py                         # Global config: paths, device
├── captioning/
│   ├── model.py                      # Loads BLIP model
│   ├── train_captioning.py          # Training script for captioning
│   ├── generate_caption.py          # Caption generation script
├── segmentation/
│   ├── visualization.py                     # Loads Mask R-CNN model
│   ├── segment.py                   # Inference on image segmentation
│   ├── train_segmentation.py        # (optional) Training script
├── app/                            # Streamlit app entry
│   ├── utils.py
├── create_directories_files.py      # for creating directories and files
├── requirements.txt                 # Python dependencies
└── README.md                        # Project overview
```

## Setup Instructions
Follow the steps below to set up and run the project:

### Step 1: Clone the Repository

```bash
git clone https://github.com/Jeet2103/image_captioning_segmentation.git
cd image_captioning_segmentation
```
###  Step 2: Set Up Python Environment (with `conda`)

```bash
conda create -n caption_env python=3.10 -y
conda activate caption_env

```
Or use `venv`:

```bash
python -m venv caption_env
source caption_env/bin/activate  # on Windows: caption_env\\Scripts\\activate

```
### Step 3: Install Dependencies

```bash
pip install -r requirements.txt

```
### Step 4: Run the Streamlit App

```bash
streamlit run app.py

```
The app will open in your default browser at `http://localhost:8501`.
