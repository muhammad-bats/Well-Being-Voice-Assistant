# Well-Being-Voice-Assistant

This repository is for a web application for a Well-being voice assistant.

## Overview


The web application chatbot can process voice commands through its frontend, 

it uses the Flask Framework for the full-stack implementation

The well-being chatbot has two versions

1. COHERE model
2. GPT-2 Fine-tuned

### COHERE 

This version uses a COHERE API model in the backbone

The model used is a simple GPT model trained on vast amounts of data. Although it is not specialized for well-being tasks, it can provide information and therapeutic responses.

### GPT-2 Fine-tuned

This version uses the GPT-2 model from the Hugging Face Transformer library.

The GPT-2 model is further fine-tuned on `CounselChat` a public dataset specialized for well-being and mental health.

`CounselChat` stores questions related to mental health and well-being and answers to these questions which licensed professional therapists provided

The `app.py` file uses further hyperparameters to control the chatbot's responses to make them more sophisticated and concise.

The `fine_tuning_bats.py` file was used for the fine-tuning process

hence ensuring professional-level responses from the model. 
