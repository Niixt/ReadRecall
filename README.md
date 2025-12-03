---
title: ReadRecall
app_file: src/app.py
sdk: gradio
sdk_version: 6.0.0
hf_oauth: true

hf_oauth_scopes:
 - inference-api
---
# ReadRecall

**ReadRecall** is an innovative application designed to enhance your reading experience by allowing you to **ask questions about the book you're currently reading without spoiling** any plot details. Utilizing advanced Large Language Models (LLM) and Retrieval-Augmented Generation (RAG) techniques, ReadRecall provides context-aware answers based on the specific chapter you are in.

## Features
- **Contextual Q&A**: Ask questions about characters, plot points, and settings relevant to your current chapter.
- **Spoiler-Free Responses**: Get answers that respect your reading progress, ensuring no spoilers.
- **LLM Integration**: Leverages powerful language models to understand and respond to your queries effectively.
- **RAG Technology**: Combines retrieval of relevant information with generation capabilities for accurate and informative answers.


## Try it on Hugging Face Spaces

[![Open in Space](https://img.shields.io/badge/Open%20in-HuggingFace%20Spaces-red?logo=HuggingFace)](https://huggingface.co/spaces/Niixt/ReadRecall)

## Installation
To install and run ReadRecall, follow these steps:

1. Create a virtual environment:
    ```bash
    conda create -n readrecall python=3.11.9 -y
    conda activate readrecall
    ```

2. Install PyTorch (**if you have a CUDA-capable GPU**):
    ```bash
    # For CUDA 11.8
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    ```

3. Install the required dependencies:
    ```bash
    pip3 install -r requirements.txt
    ```

## Usage instructions

1. Prepare your configuration file:
    
    Edit the `config.json` file to set your model preferences and paths:
    ```json
    {
        "paths": {
            // Path to store downloaded and processed books
            "documents": "src/documents",
            // Path to your Hugging Face API token
            "hf_token": "HF_TOKEN",
            // Path to your custom prompt file
            "custom_prompt": "src/utils/custom_prompt.txt"
        },
        "models": {
            // You can leave the default models or change them as you wish
        }
    }
    ```

2. For the CLI version: 

    *Recommended for faster performance*
    ```bash
    python src/app_cli.py

    # Welcome to the ReadRecall CLI!
    # Type 'exit' or 'quit' to leave the application.
    Search and load a book: [book title]

    # Ready!: 100%|███████████████████████████████████|
    # Loeaded '[book title]'. Found X chapters.
    Set chapter limit (number) leave empty for no limit: [max chapter]

    You: [your question]
    ```


3. For the Gradio web interface:
    
    *Really slower, but more user-friendly.*
    ```bash
    python src/app.py
    ```
    Then open your web browser and go to `http://127.0.0.1:7860`.

## Note

> You will need to create a file named `HF_TOKEN` in the root directory of the project containing your Hugging Face API token for model access.

> This project works only with books with free access on `archive.org`. Please test with public domain books.