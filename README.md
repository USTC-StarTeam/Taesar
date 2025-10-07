# Generative Data Transformation: From Mixed to Unified Data

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch)](https://pytorch.org/)
[![Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd)](https://hydra.cc/)
[![Weights&Biases](https://img.shields.io/badge/Weights_&_Biases-FFBE00?logo=weightsandbiases)](https://wandb.ai/)

Official implementation of **"Generative Data Transformation: From Mixed to Unified Data"**

## 📂 Project Structure

```
.
├── configs/                   # Hydra configuration files
│   ├── model/
│   │   └── SASRec.yaml
│   └── overall.yaml
├── data/                      # Data loading modules
│   ├── dataset.py
│   └── sequential_dataset.py
├── dataset/                   # Dataset processing
│   ├── raw/                   # Raw dataset files
│   ├── to_taesar.ipynb        # Data processing notebook
│   ├── to_abxi.ipynb          # Baseline data processing notebook
│   ├── to_dr4sr.ipynb         # Baseline data processing notebook
│   ├── to_cgrec.ipynb         # Baseline data processing notebook
│   └── to_syncrec.ipynb       # Baseline data processing notebook
├── model/                     # Model implementations
│   └── seq2seq_sasrec.py
├── .gitignore
├── pretrain.py                # Pretraining script
├── decoding.py                # Decoder module
├── finetune.py                # Baseline script
├── trainer.py                 # Training utilities
├── utils.py                   # Helper functions
├── run.sh                     # Example run script
├── README.md
└── requirements.txt           # Python dependencies
```


## 🚀 Quick Start

### Installation
```bash
conda env create --name Taesar --file=environments.yml
```

### Dataset Preparation
1. Download datasets from [Google Drive](https://drive.google.com/drive/folders/1b3F9FOi8X8BqUUZ0E2Aii4Ud8kEZvQbP)
```bash
cd dataset/raw/
gdown 'https://drive.google.com/uc?id=1Y7bvGSeWZ7TjGx5qA-4n59a457IpvQLO'
gdown 'https://drive.google.com/uc?id=1ogT75lYJ4fd0vNyhP1a7Kq8SC1fYBa6Y'
gdown 'https://drive.google.com/uc?id=1VJ2qx8mHi2nhyEVkoEv-3YQ5ZraG7cNs'
gdown 'https://drive.google.com/uc?id=1JdqI7sosDmqU13ZXhz1Z0rRiIOm-5hT5'
unzip Amazon_Books.zip
unzip Amazon_Electronics.zip
unzip Amazon_Sports_and_Outdoors.zip
unzip Amazon_Tools_and_Home_Improvement.zip
```
2. Process datasets using the [Jupyter notebook](dataset/to_taesar.ipynb)


### Running Experiments
```bash
for gpu_id in 0; do
    for seed in 2025; do
        python pretrain.py -m stage=run gpu_id=$gpu_id seed=$seed
        for target_dom in dom1 dom2 dom3 dom4; do
            python decoding.py -m stage=dec gpu_id=$gpu_id seed=$seed target_dom=$target_dom train_batch_size=32
        done
        for target_dom in dom1 dom2 dom3 dom4; do
            python finetune.py -m stage=tun gpu_id=$gpu_id seed=$seed train_type=new target_dom=$target_dom
            python finetune.py -m stage=tun gpu_id=$gpu_id seed=$seed train_type=sim target_dom=$target_dom
            python finetune.py -m stage=tun gpu_id=$gpu_id seed=$seed train_type=full target_dom=$target_dom
        done
    done
done
```
