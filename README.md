# Can images help recognize entities? A study of the role of images for Multimodal NER
<p align="right"><i>Authors: Shuguang Chen, Gustavo Aguilar, Leonardo Neves and Thamar Solorio</i></p> 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This repository contains the implementations of the system described in the paper ["Can images help recognize entities? A study of the role of images for Multimodal NER"]() on [The 7th Workshop on Noisy User-generated Text (W-NUT)](http://noisy-text.github.io/2021/) at the [EMNLP 2021](https://2021.emnlp.org) conference.

## Repository Structure
```
multimodal_NER
└── src
    ├── commons
    │   ├── globals.py
    │   └── utils.py
    ├── data # implementation of dataset class
    ├── modeling 
    │   ├── layers.py # implementation of neural layers
    │   ├── model.py # implementation of neural networks
    │   └── train.py # functions to build, train, and predict with a neural network
    ├── experiment.py # entire pipeline of experiments
    └── main.py # entire pipeline of our system

```

## Installation
We have updated the code to work with Python 3.9, Pytorch 1.9, and CUDA 11.2. If you use conda, you can set up the environment as follows:

```bash
conda create -n multimodal_NER python==3.8
conda activate multimodal_NER
conda install pytorch==1.9 cudatoolkit=11.2 -c pytorch
```

Also, install the dependencies specified in the requirements.txt:
```
pip install -r requirements.txt
```

## Data
In this repository, we provide some toy examples to play with the code. Due to the policy, we are not allowed to release the data. If you need, please email Shuguang Chen ([schen52@uh.edu](schen52@uh.edu)) and we will provide the following data:

```
1. original twitter text
2. associated images
3. associated image captions
```

## Running

We use config files to specify the details for every experiment (e.g., hyper-parameters, datasets, etc.). You can modify config files in the `configs` directory and run experiments with following command:

```
CUDA_VISIBLE_DEVICES=[gpu_id] python src/main.py --config /path/to/config
```

If you would like to run experiments with VisualBERT, please download the pretrained weights from [VisualBERT](https://github.com/uclanlp/visualbert/tree/master/visualbert) and replace `pretrained_weights` in the config file:

```json
    ...
    "model": {
        "name": "mner",
        "model_name_or_path": "bert-base-uncased",
        "pretrained_weights": "path/to/pretrained_weights",
        "do_lower_case": true,
        "output_attentions": false,
        "output_hidden_states": false
    },
    ...
```

## Citation
```
@misc{chen2021images,
      title={Can images help recognize entities? A study of the role of images for Multimodal NER}, 
      author={Shuguang Chen and Gustavo Aguilar and Leonardo Neves and Thamar Solorio},
      year={2021},
      eprint={2010.12712},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Contact
Feel free to get in touch via email to schen52@uh.edu.
