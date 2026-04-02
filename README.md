# NLP Homework 5: Large Language Models (LLMs)

This repository contains a comprehensive assignment focusing on the practical application of various Large Language Models (LLMs) for Natural Language Processing tasks. The project utilizes the IMDb movie reviews dataset to explore classification, generation, and prompt engineering.

## Project Structure

The project is divided into several key components:

* **`make_subset.py`**: A utility script to create a balanced subset of the IMDb dataset for efficient training and testing.
* **`bert_classification_finetuning.py`**: Implementation of fine-tuning a **BERT** (Bidirectional Encoder Representations from Transformers) model for sentiment analysis (binary classification).
* **`gpt_generation_finetuning.py`**: Fine-tuning **GPT-2** to generate movie reviews, demonstrating the generative capabilities of decoder-only architectures.
* **`flan_t5_prompt_engineering.py`**: Exploring **FLAN-T5** using prompt engineering techniques (Zero-shot) to perform sentiment classification without explicit fine-tuning.

## Key Features

* **Fine-tuning**: Hands-on experience with `Hugging Face Transformers` library to adapt pre-trained models to specific tasks.
* **Comparison**: Evaluating the performance differences between encoder-based (BERT), decoder-based (GPT), and encoder-decoder (T5) architectures.
* **Data Pipeline**: Preprocessing text data, tokenization, and managing datasets with `datasets` and `PyTorch`.

## Requirements

- Python 3.x
- PyTorch
- Transformers (Hugging Face)
- Datasets (Hugging Face)
- NumPy, Pandas

## How to Run

## How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/AhmadEgbaria1/NLP-homework5-LLM.git](https://github.com/AhmadEgbaria1/NLP-homework5-LLM.git)
Create the data subset:

Bash
python make_subset.py
Run specific tasks:

BERT classification:

Bash
python bert_classification_finetuning.py
GPT generation:

Bash
python gpt_generation_finetuning.py
FLAN-T5 prompting:

Bash
python flan_t5_prompt_engineering.py
