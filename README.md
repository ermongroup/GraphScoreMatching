# Permutation Invariant Graph Generation via Score-Based Generative Modeling

This repo contains the official implementation for the paper 

[Permutation Invariant Graph Generation via Score-Based Generative Modeling](http://proceedings.mlr.press/v108/niu20a) (AISTATS 2020),

Authors: Chenhao Niu, Yang Song, Jiaming Song, Shengjia Zhao, Aditya Grover, Stefano Ermon 

-------------------------------------------------------------------------------------
We propose a permutation invariant approach to modeling graphs, using the framework of score-based generative modeling. In particular, we design a permutation equivariant, multi-channel graph neural network to model the gradient of the data distribution at the input graph (_a.k.a_, the score function). This permutation equivariant model of gradients implicitly defines a permutation invariant distribution for graphs. We can train this graph neural network with score matching and sample from it with annealed Langevin dynamics.

## Dependencies

First, install PyTorch following the steps on its [official website](https://pytorch.org/). The code has been tested over PyTorch 1.3.1 and 1.8.1.

Then run the following command to install the other dependencies.

```shell
pip install -r requirements.txt
```

To compile the ORCA program (see http://www.biolab.si/supp/orca/orca.html) for the evaluation step, run

```shell
cd evaluation/orca && g++ -O2 -std=c++11 -o orca orca.cpp
```

## Running Experiments

### Preparing Datasets

To generate the datasets, run

```shell
mkdir data
python gen_data.py # to generate the community-small dataset
python process_dataset.py # to generate the ego-small dataset
```

### Configuring

The configurations are in the `config/` directory, written in the `YAML`  format. Refer to the comments in the given files for details. 

The output files under the directory <exp_dir>/<exp_name> (set in the `YAML` configuration file) are

```shell
.
├── config.yaml  # a copy of the configuration 
├── fig  # reconstruction of the perturbed graphs
│   └── xxx.pdf
├── info.log  # logs (if running train.py)
├── models  
│   └── xxx.pth  # the saved PyTorch checkpoint
└── sample
    ├── fig
    │   └── xxx.pdf  # images of the generated graphs
    ├── info.log  # logs (if running sampling.py)
    └── sample_data
        └── xxx.pkl  # saved python list object of the generated graphs (networkx.Graph)
```

### Training

`train.py` is the main executable file to run the whole pipeline (training, sampling, evaluation). Run `python train.py -h`  to show its usage:

``` text
usage: train.py [-h] -c CONFIG_FILE [-l LOG_LEVEL] [-m COMMENT]

Running Experiments

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG_FILE, --config_file CONFIG_FILE
                        Path of config file
  -l LOG_LEVEL, --log_level LOG_LEVEL
                        Logging Level, one of: DEBUG, INFO, WARNING, ERROR, CRITICAL
  -m COMMENT, --comment COMMENT
                        A single line comment for the experiment
```

Examples:

```shell
python train.py -c config/train_ego_small.yaml  # to run on Ego-small dataset

python train.py -c config/train_com_small.yaml  # to run on Community-small dataset
```

### Sampling 

`sample.py` is for evaluating a saved model. The usage is the same as `train.py`. To set the location of the saved model, change `model_save_dir` in the `YAML` file, e.g. `model_save_dir: 'exp/ego_small/edp-gnn_ego_small_xxx/models'`.

Examples:

```shell
python sample.py -c config/sample_ego_small.yaml  # to run on Ego-small dataset
python sample.py -c config/sample_com_small.yaml  # to run on Community-small dataset
```


