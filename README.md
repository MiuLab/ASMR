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
### Augmentation Scenario Cases
We demonstrate some scenario cases generated from our framework
<img src=./images/example.png width="100%">

## How to run
### Pretrained LLM

### Corpus


### Run the generation script

### Finetune a multimoal model


### Do-I-Demand Benchmark

#### Utterance-Only

| Model               | SBERT     |   GPT3    |
|---------------------|-----------|-----------|
| LLaVA-13B                  | 20.3  | 24.5   | 
| + place-based augmentation | 29.0  | 34.3   | 
| + action-based augmentation| 31.5  | 31.5   | 
| + both                     | 36.3  | 35.3   |
| LLaVA-13B                  | 19.5  | 22.5   | 
| + place-based augmentation | 30.3  | 36.1   |
| + action-based augmentation| 32.3  | 32.5   |
| + both                     | 34.0  | 33.8   | 

## Reference
Please cite the following paper
