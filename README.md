# Pseudocode to C++ — Transformer-Based Code Generation  

## Overview  
This project implements a **Transformer-based sequence-to-sequence model** that converts **pseudocode into executable C++ code**. The model is trained on the **SPoC dataset** and deployed as an **interactive web application** using **Streamlit**.  

## Project Highlights  
- **Dataset:** SPoC dataset, which provides paired pseudocode and C++ code.  
- **Architecture:** A **custom Transformer encoder-decoder** with token embeddings and positional encodings.  
- **Training:** Implemented in **PyTorch**, with progress tracking and model checkpointing.  
- **Deployment:** A **Streamlit UI** for real-time pseudocode-to-C++ conversion.  

## Installation  
Clone the repository and install dependencies:  

```bash
git clone https://github.com/yourusername/pseudocode-to-cpp.git
cd pseudocode-to-cpp
pip install -r requirements.txt

```
Usage
1. Train the Model
To train the model from scratch, run:

python train.py
2. Run the Streamlit App
To launch the interactive app, use:

streamlit run app.py
Dataset
The SPoC dataset consists of pseudocode and corresponding C++ implementations, used for training and evaluation.

Model Architecture
Transformer-based encoder-decoder
Custom token embeddings
Positional encodings
Trained in PyTorch
Deployment
The trained model is hosted on Hugging Face Spaces, allowing users to input pseudocode and receive generated C++ code.

Challenges and Learnings
Designing a custom tokenizer for code and pseudocode.
Handling the structural differences between natural language and programming syntax.
Efficient checkpointing for resuming training.
Future Improvements
Train on additional datasets to improve accuracy.
Implement beam search decoding for better output quality.
Extend support for other programming languages.

Read the full article: https://medium.com/@sultanularfeen/pseudocode-to-c-code-building-a-transformer-model-from-scratch-57b1ba3bab58
