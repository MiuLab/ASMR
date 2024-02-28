ASMR: Augmenting Life Scenario using Large Generative Models for Robotic Action Reflection
===
[IWSDS 2024 paper]

## Overview

## Framework
<img src=./images/augmentation.png width="100%">

## Requirements
* Python >= 3.6
* Transformers
* torch

## Datasets
Our generated dataset can be found in the [google drive]()

## How to run
### Pretrained LLM

### Corpus


### Run the generation script

### Finetune a multimoal model


### Do-I-Demand Benchmark

|                     |    Utterance-Only  | Description + Utterance    |
|---------------------|--------------------|----------------------------|
| Model               | SBERT     |   GPT3 |     SBERT    |     GPT3    |
|---------------------|-----------|--------|--------------|-------------|
| LLaVA-13B                  | 20.3  | 24.5   | 28.3      | 34.8        |
| + place-based augmentation | 29.0  | 34.3   | 33.3      | 39.0        |
| + action-based augmentation| 31.5  | 31.5   | 45.5      | 45.5        |
| + both                     | 36.3  | 35.3   | 48.5      | 47.8        |

## Reference
Please cite the following paper
