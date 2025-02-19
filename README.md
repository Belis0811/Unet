# AI-based architecture: Self-Attention UNet + Transformer for Cell Viability & Movement Prediction
This repository contains an **AI-based framwork** that segments live vs dead cells from **quantitative phase imaging (QPI) data**
and forcast their future movement. The framework combines a **self-attention UNet**(for segmentation) with a **transformer**(for dynamic cell tracking),
building upon our paper:
>[AI-enabled live-dead cell viability classification and
motion forecasting](https://arxiv.org/pdf/2312.13311)
>
>Anzhe Cheng, Chenzhong Yin, Michael A.S. Lamba etal.

## Overview
Paper is implemented with official pytorch
![Overview Image](docs/figures/overview.png?raw=true "Overview workflow of the proposed architecture")

This illustrates a high‐level overview of our self‐attention UNet + transformer pipeline. The diagram shows how **unlabeled CHO cell images** from
**SLIM** are processed by our segmentation model and then used for cell viability classification and future movement prediction.

## Repository Structrure
```
Unet/
├─ .vscode/                     # Editor/IDE settings
├─ data/
│  ├─ processed/                # Data processed by process_data.ipynb
│  │  ├─ X_train.npy
│  │  └─ Y_train.npy
│  ├─ result/
│  │  ├─ test/                  #Test img
│  │  └─ val/                   # Validation img
│  └─ sample_data/
│     ├─ 10x_magnification/   
│     │  ├─ f0_t0_i0_ch0_c0_r0_z0/
│     │  │  ├─ images/
│     │  │  │  └─ f0_t0_i0_ch0_c0_r0_z0_mSLIM.png
│     │  │  └─ masks/
│     │  │     ├─ f0_t0_i0_ch1_c0_r0_z0_m49 DAPI.png
│     │  │     └─ f0_t0_i0_ch2_c0_r0_z0_m38 Green Fluorescent Prot.png
│     │  ├─ f0_t0_i1_ch0_c0_r0_z0/
│     │  │  ├─ images/
│     │  │  └─ masks/
│     │  └─ ... (other subfolders for 10× data)
│     ├─ 40x_magnification/              #Similar Structure with 10x
│     │  ├─ f0_t0_i0_ch0_c0_r0_z0/
│     │  │  ├─ images/
│     │  │  └─ masks/
│     │  └─ ... (other subfolders for 40× data)
├─ docs/
│  ├─ figures/                
│  │  ├─ overview.png
│  │  └─ segmentation_metrics.png
│  └─ paper/
│     └─ Cell_Segmentation_....pdf
├─ notebooks/
│  ├─ live_dead_segment.ipynb       # UNet segmentation training
│  ├─ process_data.ipynb            # Data loading/preprocessing
│  └─ Unet_Transformer_move.ipynb   # Transformer for cell movement
├─ weights/
│  ├─ checkpoint_informer.pth
│  ├─ checkpoint_unet.pth
│  ├─ checkpoint.pth
│  └─ saved_weight_folder.txt
├─ LICENSE
├─ SECURITY.md
├─ requirements.txt
├─ .gitignore
└─ README.md

```
## Requirements
* **Python** 3.8+
* **PyTorch** 2.0.0+
* **torchvision** 0.15.0+
*  **NumPy**,**scikit-image**,**matplotlib**,etc
  
Please go to the `requirement.txt` file to check all dependencies.

Or run the following code to install:
```
pip install -r requirements.txt
```
## Data Processing
First, make sure you are at the correct working directory. Then direct to the `notebooks` folder by
```
cd notebooks
```

run every cell in the `process_data.ipynb` file, which includes image loading, processing, and data splitting. 

*Note that we use **Spatial Light Interference Microscopy(SLIM)** images of unlabeled CHO cells at different magnifications (10x,40x), which can
be find at the directory: `Unet/data/sample_data`. By default we will process 40x data, if you would like to process 10x data, pleaes change the 
TRAIN_PATH and TEST_PATH at the first cell to the following:*
```
TRAIN_PATH = os.path.abspath("../data/sample_data/10x_magnification") 
TEST_PATH = os.path.abspath("../data/sample_data/10x_magnification") 
```

After running the last cell of this file,
you will generate two files called `X_train.npy` and `Y-train.npy`, located at the folder: `./Unet/data/processed`


## Training Segmentation Model

Then, navigate to the file called `live_dead_segment.ipynb`, which trains the **self-attention UNet** to segment live vs. dead cells.

*Note that you have the option to not re-train the model from scratch, please follow the comments in the file to decide of using checkpoint file*

## Training Movement Prediction Model

Now we want to use the transformer module to predict future cell positions/time-steps to finally combine them and make what we called AI-based architecture.

To do so, we navigate to the file called `Unet_Transformer_move.ipynb`and run all cells.

## Results

![alt text](docs/figures/Segmentation_Comparison.png?raw=true "Error rate of different methods")

Our Self-attention UNet is outperfoming SAM2 and E-U-Net at various matrices. Specifically,
the model correctly identified **99.9%** of the live cells, while only **0.1%** of live cells were misclassified as dead. Similarly,
**99.9%** of the dead cells were correctly identified, with only **0.1%** of dead cells misclassified as live.The precision, recall, and
F1 scores for both live and dead cells are all above **98%** as illustrated by the table below. 

![alt text](docs/figures/segmentation_metrics.png?raw=true "Error rate of different methods")

Also, with regarding the performance of Dynamic Cell Tracking, the graph below demonstrates how predicted cell trajectories closely track actual observed movements, with only minor deviations in
some instances.

![alt text](docs/figures/cell_move_predict.png?raw=true "Error rate of different methods")
## Acknowledgements

* This work was supported by []
* Special thanks to Professor Paul Bogdan, Professor Shahin Nazarian, Professor Nicholas Kotov and Professor Dan Ionascu for their support!

## Citation

Please cite the following paper if this model assists your research:
```bibtex
@inproceedings{cheng2024ai,
  title={AI-enabled live-dead cell viability classification and motion forecasting},
  author={Cheng, Anzhe and Yin, Chenzhong and Lamba, Michael A.S. and Sertorio, Mathieu and DeJesus, Maldonado and Alexis, Jorge and Sathler, Alexandre R. and Chang, Yu and Chiritescu, Catalin and Best, Catherine A. and Ionascu, Dan and Kotov, Nicholas and Nazarian, Shahin and Bogdan, Paul},
  booktitle={TBD},
  pages={TBD},
  year={2025},
  organization={TBD}
}
```

## Final Noates
* Correspondence and requests for materials should be addressed to Professor Paul Bogdan (email: pbogdan@usc.edu)
