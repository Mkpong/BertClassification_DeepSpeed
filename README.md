# BertClassification_DeepSpeed
## Prerequisites
- Ubuntu : 20.04.6 LTS
- Python : v3.9.11
- Anaconda3 : v23.7.4
---
## How to use
**1. Create conda env**
```
conda create --name DeepSpeed python=3.9.11
```
**2. Install pytorch**
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
**3. Install pip packages**
```
pip install -r requirements
```
**4. Run DeepSpeed**
-> You can run Data,Tensor,Pipeline Parallel train code in DDP, DTP, DPP folder
```
cd DDP #DTP, DPP
deepspeed --num_gpus=2 ../train.py
```
