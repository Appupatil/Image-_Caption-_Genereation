# AI Image Caption Generator

An AI-powered application that generates captions for images using Vision Transformer and GPT-2.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
streamlit run main.py
```

## Features
- Upload images (JPG, JPEG, PNG)
- AI-generated captions
- Adjustable caption length and beam search

## Model
- Vision Transformer (ViT) + GPT-2
- Model: nlpconnect/vit-gpt2-image-captioning
```

### Create a **`.gitignore`** file:
```
__pycache__/
*.pyc
.env
*.log
.streamlit/
