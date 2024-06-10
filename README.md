# MXfold2
RNA secondary structure prediction using deep learning with thermodynamic integration

## Installation

### System requirements
* python (>=3.7)
* pytorch (>=1.4)
* C++17 compatible compiler (tested on Apple clang version 12.0.0 and GCC version 7.4.0) (optional)

### Install from wheel

We provide the wheel python packages for several platforms at [the release](https://github.com/mxfold/mxfold2/releases). You can download an appropriate package and install it as follows:

    % pip3 install mxfold2-0.1.2-cp310-cp310-manylinux_2_17_x86_64.whl

### Install from sdist

You can build and install from the source distribution downloaded from [the release](https://github.com/mxfold/mxfold2/releases) as follows:

    % pip3 install mxfold2-0.1.2.tar.gz

To build MXfold2 from the source distribution, you need a C++17 compatible compiler.

## Prediction

You can predict RNA secondary structures of given FASTA-formatted RNA sequences like:

    % mxfold2 predict test.fa
    >DS4440
    GGAUGGAUGUCUGAGCGGUUGAAAGAGUCGGUCUUGAAAACCGAAGUAUUGAUAGGAAUACCGGGGGUUCGAAUCCCUCUCCAUCCG
    (((((((........(((((..((((.....))))...)))))...................(((((.......)))))))))))). (24.8)

By default, MXfold2 employs the parameters trained from TrainSetA and TrainSetB (see our paper).

We provide other pre-trained models used in our paper. You can download [``models-0.1.0.tar.gz``](https://github.com/mxfold/mxfold2/releases/download/v0.1.0/models-0.1.0.tar.gz) and extract the pre-trained models from it as follows:

    % tar -zxvf models-0.1.0.tar.gz

Then, you can predict RNA secondary structures of given FASTA-formatted RNA sequences like:

    % mxfold2 predict @./models/TrainSetA.conf test.fa
    >DS4440
    GGAUGGAUGUCUGAGCGGUUGAAAGAGUCGGUCUUGAAAACCGAAGUAUUGAUAGGAAUACCGGGGGUUCGAAUCCCUCUCCAUCCG
    (((((((.((....))...........(((((.......))))).(((((......))))).(((((.......)))))))))))). (24.3)

Here, ``./models/TrainSetA.conf`` specifies a lot of parameters including hyper-parameters of DNN models.

## Training

MXfold2 can train its parameters from BPSEQ-formatted RNA sequences. You can also download the datasets used in our paper at [the release](https://github.com/mxfold/mxfold2/releases/tag/v0.1.0). 

    % mxfold2 train --model MixC --param model.pth --save-config model.conf data/TrainSetA.lst

You can specify a lot of model's hyper-parameters. See ``mxfold2 train --help``. In this example, the model's hyper-parameters and the trained parameters are saved in ``model.conf`` and ``model.pth``, respectively.

## Web server

A web server is working at http://www.dna.bio.keio.ac.jp/mxfold2/.


## References

* Sato, K., Akiyama, M., Sakakibara, Y.: RNA secondary structure prediction using deep learning with thermodynamic integration. *Nat Commun* **12**, 941 (2021). https://doi.org/10.1038/s41467-021-21194-4



## Use RNA-SDB dataset

### Setup


```bash
mamba create -n mxfold2-train python=3.9 ipython
mamba activate mxfold2-train

mamba install numpy pandas pyarrow huggingface_hub pybind11 pytorch==1.13.1 torchvision pytorch-cuda=11.7 tqdm wandb wheel -c pytorch -c nvidia
pip install .
```

To use RNA-SDB dataset hosted on Huggingface-Hub, login to your huggingface account:

```bash
huggingface-cli login 
```





### Training

We've updated training script to use new dataset wrapper `RnaSdbDataset` to load parquet format (with cols: `seq_id`, `seq`, `db_structure`),
consistent with RNA-SDB format.

> [!CAUTION]
> Replace `DATASET` with the actual rna-sdb dataset (right now it is a toy dataset in Alice's HF repo).

To train model on RNA-SDB dataset hosted on huggingface-hub:

```bash
export WDIR=wkdir/run_test
export DATASET="AliceGao/test1/split_1_cache_test.1000.pq"
export ENTITY=my-wandb-entity    # replace with yours
export PROJECT=rna-sdb-mxfold2-train  # replace with yours
export GROUP=my-wandb-group  # replace with yours
export JOBTYPE=my-wandb-job-type  # replace with yours
export SPLITNAME=my-split-1  # replace with yours
mkdir -p $WDIR
CUDA_VISIBLE_DEVICES=0 mxfold2 train --model MixC --gpu 0 --log-dir $WDIR --epochs 1 \
 --entity $ENTITY --project $PROJECT --group debug --job_type $JOBTYPE --split_name $SPLITNAME \
 $DATASET
```

At the end of training, the checkpoint `$WDIR/checkpoint.pt` is exported,
which also gets uploaded to wandb as artifact (with id in the format of `$ENTITY/$PROJECT/mxfold2-model:v*`).


### Inference

> [!CAUTION]
> Replace `DATASET` with the actual rna-sdb dataset (test-split) (right now it is a toy dataset in Alice's HF repo).


Run inference on RNA-SDB dataset, using the model trained from above:


```bash
export WDIR=wkdir/run_test
export DATASET="AliceGao/test1/split_1_cache_test.1000.pq"
export ENTITY=my-wandb-entity    # replace with yours
export PROJECT=rna-sdb-mxfold2-train  # replace with yours
export GROUP=my-wandb-group  # replace with yours
export JOBTYPE=my-wandb-job-type  # replace with yours
export SPLITNAME=my-split-1  # replace with yours
CUDA_VISIBLE_DEVICES=0 mxfold2 predict --model MixC --param $WDIR/checkpoint.pt --gpu 0 --bpp $WDIR/prediction.pq \
  --entity $ENTITY --project $PROJECT --group debug --job_type $JOBTYPE --split_name $SPLITNAME \
  $DATASET
```

Note that the local file path to checkpoint `$WDIR/checkpoint.pt` can also be replaced by wandb artifact ID (as produced from training).

The above inference script writes to a dataframe (in parquet format) with 3 columns: 

    - `seq`: sequence

    - `bpseq`: bpseq-like list (length `len(seq)`) of 3-tuples, e.g. `[[1, 'A', 0], [2, 'A', 0], [3, 'G', 0], [4, 'A', 0], ...]`  (note that bpseq indices are 1-based)

    - `bp_matrix`: 2D matrix computed by MXFOLD2, with shape `(len(seq)+1) x (len(seq)+1)`

The parquet dataframe is also uploaded to wandb as artifact (with id in the format of `$ENTITY/$PROJECT/mxfold2-prediction:v*`).
