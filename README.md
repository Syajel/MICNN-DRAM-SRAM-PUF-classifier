# Classification of DRAM and SRAM PUF Responses Using a Multi-Input Convolutional Neural Network

This repository contains the implementation of a Multi-Input Convolutional Neural Network (CNN) designed to classify DRAM and SRAM PUF responses under intact, corrupted, and noisy conditions. The project is based on the research presented in my 2025 Master's thesis.

---

## Overview

This work addresses the problem of robust memory-based PUF classification.

A Multi-Input CNN model is used:
- One branch processes image-formatted PUF responses (A separate branch for each, DRAM and SRAM).
- The other branch processes metadata (e.g. address range, time delay).
- Branches are fused at the fully connected layer level.
- The response is classified into one of the 146 classes

---

## Project Structure

.
├── data/               # Input DRAM and SRAM response files
│   ├── dram/			
│   	 ├── raw/			# DRAM raw dataset
│   	 ├── train/		    # DRAM training data
│  		 └── test/      	# DRAM test data
│   └── sram/
│   	 ├── raw/			# SRAM raw dataset
│   	 ├── train/		    # SRAM training data
│  		 └── test/      	# SRAM test data
├── models/             # Trained model weights
├── plots/              # Evaluation figures
├── src/
│   ├── augment_dram_data.py	# DRAM augmentation script
│   ├── augment_sram_data.py	# SRAM augmentation script
│   ├── split_data.py        	# Data compression and splitting into train and test folders script
│   ├── plot_results.py     	# Graph plotting script
│   ├── load_datasets.py 		# Dataset loader
│   ├── model.py        		# CNN Model
│   ├── train.py        		# Training script
│   ├── test.py         		# Evaluation script
│   └── __init__.py        	
├── requirements.txt    # Python dependencies
└── README.md           # This file

---

## Setup Instructions

### 1. Clone the repository

git clone https://github.com/Syajel/MICNN-DRAM-SRAM-PUF-classifier.git
cd multi-input-puf-classifier

### 2. Create a virtual environment

python -m venv venv

### 3. Activate the environment

* On **Linux/macOS**:

source venv/bin/activate

* On **Windows**:

venv\Scripts\activate

### 4. Install dependencies inside the environment

pip install -r requirements.txt

### 5. (Optional) Deactivate when done

deactivate


---

## Usage

### Move datasets to directory "./data/DRAM/raw" and "./data/SRAM/raw"

### Compress and split the raw responses into train and test datasets

python -m src.split_data

### Set noise probabilities in "augment_dram_data.py" augment DRAM data

python -m src.augment_dram_data

### Augment SRAM data

python -m src.augment_sram_data

### Train the model

python -m src.train

### Evaluate the model

python -m src.test


## Features

* DRAM and SRAM classification
* Support for intact, corrupted, and noisy input
* Probabilistic asymmetric noise generation
* Metadata incorporation
* Output visualization (accuracy/loss curves)

---

## Citation

If you use this code in your research, please cite:

Shamel Khalil Mohammady Khalil, “Classification of DRAM and SRAM PUF Responses Using a Multi-Input Convolutional Neural Network” Master’s Thesis, 2025.

---

## Contact

If you have any questions, feel free to reach out via [shamel.khalil@gmail.com].

