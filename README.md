# AutoTool

### 1. Introduction

This repository contains the implementation of the paper **[Are Tools Always Beneficial? Learning to Invoke Tools Adaptively for Dual-Mode Multimodal LLM Reasoning](https://arxiv.org/pdf/2605.19852)**
> *In Forty-Third International Conference on Machine Learning (ICML), 2026*

### 2. Dataset Construction

The training data is derived from the **[DeepEyes-Datasets-47k](https://huggingface.co/datasets/ChenShawn/DeepEyes-Datasets-47k)** dataset. The evaluation benchmark covers multiple publicly available datasets, including perception, localization, mathematical reasoning, and hallucination evaluation tasks. All datasets are accessible via Hugging Face.

### 3. Train

We provide a training script for Qwen2.5-VL-7B in `autoTool_qwen_2_5_7B_nonbf60.sh`.

### 4. Test

The evaluation code is located in the `eval/` directory.

### 5. Acknowledgement

This project is based on the code from the [VERL](https://github.com/verl-project/verl) and [DeepEyes](https://github.com/Visual-Agent/DeepEyes) project.

Thanks a lot for their great works.
