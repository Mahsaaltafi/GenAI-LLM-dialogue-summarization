# GenAI-LLM-Dialogue-Summarization  

This repository contains notebooks for training and evaluating generative AI models for dialogue summarization using large language models (LLMs). It demonstrates both full fine-tuning and parameter-efficient fine-tuning (PEFT/LoRA) on the [FLAN-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5) model, with evaluation using ROUGE metrics.  

## Contents  
- **train_summarizer.ipynb** — fine-tunes FLAN-T5 on the dialogue summarization task  
- **train_detox_summarizer.ipynb** — trains a toxicity-aware summarizer using reinforcement learning (PPO) with a reward model  
- **inference_summarizer.ipynb** — runs inference and compares zero-shot, full fine-tuned, and PEFT/LoRA-adapted models  

## Features  
- Zero-shot, one-shot, and few-shot inference experiments  
- Full model fine-tuning with Hugging Face `Trainer`  
- Parameter-efficient fine-tuning with LoRA adapters  
- Reinforcement learning with Proximal Policy Optimization (PPO) for detoxification  
- Evaluation with ROUGE metrics and qualitative comparison  

## Requirements  
- Python 3.8+  
- [Hugging Face Transformers](https://github.com/huggingface/transformers)  
- Datasets, Evaluate, PEFT, Torch, TensorFlow (see notebooks for exact versions)  

Install dependencies with:  

```bash
pip install -r requirements.txt
