# ReadRecall

**ReadRecall** is an innovative application designed to enhance your reading experience by allowing you to **ask questions about the book you're currently reading without spoiling** any plot details. Utilizing advanced Large Language Models (LLM) and Retrieval-Augmented Generation (RAG) techniques, ReadRecall provides context-aware answers based on the specific chapter you are in.

## Features
- **Contextual Q&A**: Ask questions about characters, plot points, and settings relevant to your current chapter.
- **Spoiler-Free Responses**: Get answers that respect your reading progress, ensuring no spoilers.
- **LLM Integration**: Leverages powerful language models to understand and respond to your queries effectively.
- **RAG Technology**: Combines retrieval of relevant information with generation capabilities for accurate and informative answers.

## Installation
To install and run ReadRecall, follow these steps:

1. Create a virtual environment:
    ```bash
    conda create -n readrecall python=3.11.9 -y
    conda activate readrecall
    ```

2. Install PyTorch (if you have a CUDA-capable GPU, otherwise install the CPU version):
    ```bash
    # For CUDA 11.8
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    # For CPU only
    # pip3 install torch torchvision
    ```

3. Install the required dependencies:
    ```bash
    pip3 install -r requirements.txt
    ```
