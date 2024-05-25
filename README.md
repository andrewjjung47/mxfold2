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



## Added by Alice

### Setup


```bash
mamba create -n mxfold2-train python=3.9 ipython

mamba activate mxfold2-train

mamba install numpy pandas fastparquet pybind11 pytorch==1.13.1 torchvision pytorch-cuda=11.7 tqdm wheel -c pytorch -c nvidia
# mamba install numpy pandas fastparquet pybind11 pytorch==1.13.1 torchvision tqdm wheel -c pytorch   # no GPU

pip install .
# for dev: pip install --editable .
```


:exclamation: Note that if working off GPU server and using `truenas` the dev-mode (`pip install --editable .`) doesn't seem to work. Instead do `pip install .`.

```bash
Traceback (most recent call last):
  File "/home/alice/conda/envs/mxfold2-train/bin/mxfold2", line 5, in <module>
    from mxfold2.__main__ import main
  File "/mnt/dg_shared_truenas/for_alice/work/mxfold2/mxfold2/__main__.py", line 6, in <module>
    from .predict import Predict
  File "/mnt/dg_shared_truenas/for_alice/work/mxfold2/mxfold2/predict.py", line 14, in <module>
    from .fold.mix import MixedFold
  File "/mnt/dg_shared_truenas/for_alice/work/mxfold2/mxfold2/fold/mix.py", line 2, in <module>
    from .. import interface
ImportError: /mnt/dg_shared_truenas/for_alice/work/mxfold2/mxfold2/interface.cpython-39-x86_64-linux-gnu.so: failed to map segment from shared object
```





<!-- test list of files:

```bash
mkdir wkdir
ls -a1 ../rna_sdb/datasets/bpRNA/bprna/TS0/bpRNA_RFAM_15* | xargs realpath > wkdir/bprna_tr0_small.lst

mkdir wkdir/log
mxfold2 train --model MixC --param wkdir/log/model.pth --save-config wkdir/log/model.conf \
--log-dir wkdir/log/  \
 --epochs 2  wkdir/bprna_tr0_small.lst
``` -->





### Training

- updated code to only save the last checkpoint, it'll be named `checkpoint.pt`

- added new dataset wrapper `RnaSdbDataset` to load parquet format (with cols: `seq_id`, `seq`, `db_structure`),
replaced the default dataset with this one


Training model on pq dataset (GPU):


```bash
mkdir -p wkdir/run_1/
CUDA_VISIBLE_DEVICES=0 mxfold2 train --model MixC --param wkdir/run_1/model.pth --save-config wkdir/run_1/model.conf --gpu 0 --log-dir wkdir/run_1/  --epochs 10 --train_max_len 1000 /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/rna_sdb/split_1_cache_train.pq
```

Training model on pq dataset (CPU-test):

```bash
mkdir -p wkdir/run_0/
mxfold2 train --model MixC --param wkdir/run_0/model.pth --save-config wkdir/run_0/model.conf \
--log-dir wkdir/run_0/  \
 --epochs 2  wkdir/split_3_cache_test.pq
```

TODO slow for long sequences?


### Inference


Inference on pq dataset:


```bash
CUDA_VISIBLE_DEVICES=0 mxfold2 predict --model MixC --param wkdir/debug_1/checkpoint.pt --gpu 0 --bpp wkdir/debug_1/prediction.pq /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/rna_sdb/split_3_cache_test.pq
```





### Debug notes



debug run on small test set:


```bash
CUDA_VISIBLE_DEVICES=0 mxfold2 train --model MixC --param wkdir/debug_1/model.pth --save-config wkdir/debug_1/model.conf --gpu 0 --log-dir wkdir/debug_1/  --epochs 1 --train_max_len 100 /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/rna_sdb/split_3_cache_test.pq
```

This generates `wkdir/debug_1/checkpoint.pt` for debug use.