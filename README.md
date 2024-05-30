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

mamba install numpy pandas pyarrow pybind11 pytorch==1.13.1 torchvision pytorch-cuda=11.7 tqdm wandb wheel -c pytorch -c nvidia
# mamba install numpy pandas fastparquet pybind11 pytorch==1.13.1 torchvision pytorch-cuda=11.7 tqdm wandb wheel -c pytorch -c nvidia
# mamba install numpy pandas fastparquet pybind11 pytorch==1.13.1 torchvision tqdm wheel -c pytorch   # no GPU

pip install .
# for dev: pip install --editable .
```

:exclamation:  Make sure to install `pyarrow` not `fastparquet`, so that we can load Andrew's `pq` dataset. 


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


Inference on pq dataset (set `max_num` to process a small number of seqs for debugging):


```bash
CUDA_VISIBLE_DEVICES=0 mxfold2 predict --model MixC --param wkdir/debug_1/checkpoint.pt --gpu 0 --bpp wkdir/debug_1/prediction.pq --max_num 10 /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/rna_sdb/split_3_cache_test.pq
```


- writes to a df with 3 cols: 

    - `seq`: just the sequence

    - `bpseq`: bpseq-like list (length `len(seq)`) of 3-tuples, something like: `[[1, 'A', 0], [2, 'A', 0], [3, 'G', 0], [4, 'A', 0], ...]`  (note that bpseq indices are 1-based)

    - `bp_matrix`: 2D matrix returned by their script, I'm still not sure what this is (did not dive deep into their inference code) 
    (note that the shape is `(len(seq)+1) x (len(seq)+1)`)



### Debug notes


#### 2024-05-25

debug run on small test set:


```bash
CUDA_VISIBLE_DEVICES=0 mxfold2 train --model MixC --param wkdir/debug_1/model.pth --save-config wkdir/debug_1/model.conf --gpu 0 --log-dir wkdir/debug_1/  --epochs 1 --train_max_len 100 /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/rna_sdb/split_3_cache_test.pq
```

This generates `wkdir/debug_1/checkpoint.pt` for debug use.


Run inference:


```bash
CUDA_VISIBLE_DEVICES=0 mxfold2 predict --model MixC --param wkdir/debug_1/checkpoint.pt --gpu 0 --bpp wkdir/debug_1/prediction.pq --max_num 10 /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/rna_sdb/split_3_cache_test.pq
```



Inspect ouptut:

```python
import numpy as np
import pandas as pd

df = pd.read_parquet('wkdir/debug_1/prediction.pq')
seq = df.iloc[0]['seq']
x = np.array(df.iloc[0]['bp_matrix'])
bpseq = df.iloc[0]['bpseq']

print(len(seq), len(bpseq), x.shape)  # 217 217 (218, 218)

bpseq[:4]  # [[1, 'A', 0], [2, 'A', 0], [3, 'G', 0], [4, 'A', 0]]

np.sum(x)  # 0.024204140787577863  # Hmm...
```



Added wandb logging (hard-coded to my account, not uploading model artifact). 
See if we can overfit on small test set by training for more epochs:


```bash
WANDB_API_KEY=fc31024445b1bd60765be0295fb8f4a8ca0b389e  CUDA_VISIBLE_DEVICES=0 mxfold2 train --model MixC --param wkdir/debug_2/model.pth --save-config wkdir/debug_2/model.conf --gpu 0 --log-dir wkdir/debug_2/  --epochs 10 --train_max_len 500 /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/rna_sdb/split_3_cache_test.pq
```

Some long sequences are really slow: https://wandb.ai/psi-lab/rna-sdb-mxfold2-train/runs/5gyi0a0g

Set max len to 100 and rerun:


```bash
WANDB_API_KEY=fc31024445b1bd60765be0295fb8f4a8ca0b389e  CUDA_VISIBLE_DEVICES=0 mxfold2 train --model MixC --param wkdir/debug_2/model.pth --save-config wkdir/debug_2/model.conf --gpu 0 --log-dir wkdir/debug_2/  --epochs 10 --train_max_len 100 /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/rna_sdb/split_3_cache_test.pq
```


Why is this not improving? loss looks weird... shall we train their training set and run from master branch?




check `wkdir/debug_2`: TODO



#### 2024-05-26


```bash
WANDB_API_KEY=fc31024445b1bd60765be0295fb8f4a8ca0b389e  CUDA_VISIBLE_DEVICES=0 mxfold2 train --model MixC --param wkdir/debug_3/model.pth --save-config wkdir/debug_3/model.conf --gpu 0 --log-dir wkdir/debug_3/  --epochs 10 --train_max_len 100 /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/bpRNA/bprna_for_alice.pq
```


inference:



```bash
CUDA_VISIBLE_DEVICES=0 mxfold2 predict --model MixC --param wkdir/debug_3/checkpoint.pt --gpu 0 --bpp wkdir/debug_3/prediction.pq --max_num 50 /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/bpRNA/bprna_for_alice.pq
```



update data loader to return torch tensor:


```bash
WANDB_API_KEY=fc31024445b1bd60765be0295fb8f4a8ca0b389e  CUDA_VISIBLE_DEVICES=0 mxfold2 train --model MixC --param wkdir/debug_4/model.pth --save-config wkdir/debug_4/model.conf --gpu 0 --log-dir wkdir/debug_4/  --epochs 2 --train_max_len 100 /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/bpRNA/bprna_for_alice.pq
```

run: https://wandb.ai/psi-lab/rna-sdb-mxfold2-train/runs/c09a4s98, 
loss scale seems to be more consistent with what was seen from their master branch!


```bash
CUDA_VISIBLE_DEVICES=0 mxfold2 predict --model MixC --param wkdir/debug_4/checkpoint.pt --gpu 0 --bpp wkdir/debug_4/prediction.pq --max_num 50 /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/bpRNA/bprna_for_alice.pq
```

train on full bprna without length limit:


```bash
WANDB_API_KEY=fc31024445b1bd60765be0295fb8f4a8ca0b389e  CUDA_VISIBLE_DEVICES=0 mxfold2 train --model MixC --param wkdir/debug_5/model.pth --save-config wkdir/debug_5/model.conf --gpu 0 --log-dir wkdir/debug_5/  --epochs 10  /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/bpRNA/bprna_for_alice_train.pq
```

Adding wandb: model artifact, inference. Train a model:

```bash
WANDB_API_KEY=fc31024445b1bd60765be0295fb8f4a8ca0b389e  CUDA_VISIBLE_DEVICES=0 mxfold2 train --model MixC --param wkdir/debug_6/model.pth --save-config wkdir/debug_6/model.conf --gpu 0 --log-dir wkdir/debug_6/  --epochs 1 --train_max_len 50 /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/bpRNA/bprna_for_alice_train.pq
```

Check wandb model artifact: `psi-lab/rna-sdb-mxfold2-train/mxfold2-model:v0`. update inference to use wandb artifact. Check predict script:



```bash
WANDB_API_KEY=fc31024445b1bd60765be0295fb8f4a8ca0b389e  CUDA_VISIBLE_DEVICES=0 mxfold2 predict --model MixC --param psi-lab/rna-sdb-mxfold2-train/mxfold2-model:v0 --gpu 0 --bpp wkdir/debug_6/prediction.pq --max_num 50 /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/bpRNA/bprna_for_alice.pq
```

Inspect predictions: looks good. Update inference to upload predictions to wandb, re-run above:

```bash
WANDB_API_KEY=fc31024445b1bd60765be0295fb8f4a8ca0b389e  CUDA_VISIBLE_DEVICES=0 mxfold2 predict --model MixC --param psi-lab/rna-sdb-mxfold2-train/mxfold2-model:v0 --gpu 0 --bpp wkdir/debug_6/prediction.pq --max_num 50 /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/bpRNA/bprna_for_alice.pq
```

Check wandb model artifact: `psi-lab/rna-sdb-mxfold2-predict/mxfold2-prediction:v0` looks ok.




### Experiments on rna-sdb dataset


From Andrew:

```
I uploaded the datasets to the drive I've been using: /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/rna_sdb

For the training, you can use mxfold2_rnasdb_split_*_train.pq. They should all have 107,840 samples, which is somewhat equivalent to 10 epoch over bpRNA. As we discussed, you only need to run one epoch over the entire dataset.

For the testing, you can use mxfold2_archiveII_split_*.pq but please make sure to use the corresponding split as the dataset split the model was trained on (i.e. keep split_1 training and testing together)
```


Training (each split on one GPU):


```bash
mamba activate mxfold2-train
cd /mnt/dg_shared_truenas/for_alice/work/mxfold2
export SPLIT_ID=1
WANDB_API_KEY=fc31024445b1bd60765be0295fb8f4a8ca0b389e  CUDA_VISIBLE_DEVICES=0 mxfold2 train --model MixC --param wkdir/rna_sdb/split_${SPLIT_ID}/model.pth --save-config wkdir/rna_sdb/split_${SPLIT_ID}/model.conf --gpu 0 --log-dir wkdir/rna_sdb/split_${SPLIT_ID}/  --epochs 1  /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/rna_sdb/mxfold2_rnasdb_split_${SPLIT_ID}_train.pq
# https://wandb.ai/psi-lab/rna-sdb-mxfold2-train/runs/sbs7l1z8

mamba activate mxfold2-train
cd /mnt/dg_shared_truenas/for_alice/work/mxfold2
export SPLIT_ID=2
WANDB_API_KEY=fc31024445b1bd60765be0295fb8f4a8ca0b389e  CUDA_VISIBLE_DEVICES=1 mxfold2 train --model MixC --param wkdir/rna_sdb/split_${SPLIT_ID}/model.pth --save-config wkdir/rna_sdb/split_${SPLIT_ID}/model.conf --gpu 0 --log-dir wkdir/rna_sdb/split_${SPLIT_ID}/  --epochs 1  /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/rna_sdb/mxfold2_rnasdb_split_${SPLIT_ID}_train.pq
# https://wandb.ai/psi-lab/rna-sdb-mxfold2-train/runs/dflbdnoc


mamba activate mxfold2-train
cd /mnt/dg_shared_truenas/for_alice/work/mxfold2
export SPLIT_ID=3
WANDB_API_KEY=fc31024445b1bd60765be0295fb8f4a8ca0b389e  CUDA_VISIBLE_DEVICES=2 mxfold2 train --model MixC --param wkdir/rna_sdb/split_${SPLIT_ID}/model.pth --save-config wkdir/rna_sdb/split_${SPLIT_ID}/model.conf --gpu 0 --log-dir wkdir/rna_sdb/split_${SPLIT_ID}/  --epochs 1  /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/rna_sdb/mxfold2_rnasdb_split_${SPLIT_ID}_train.pq
# https://wandb.ai/psi-lab/rna-sdb-mxfold2-train/runs/mv7a5oam

mamba activate mxfold2-train
cd /mnt/dg_shared_truenas/for_alice/work/mxfold2
export SPLIT_ID=4
WANDB_API_KEY=fc31024445b1bd60765be0295fb8f4a8ca0b389e  CUDA_VISIBLE_DEVICES=3 mxfold2 train --model MixC --param wkdir/rna_sdb/split_${SPLIT_ID}/model.pth --save-config wkdir/rna_sdb/split_${SPLIT_ID}/model.conf --gpu 0 --log-dir wkdir/rna_sdb/split_${SPLIT_ID}/  --epochs 1  /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/rna_sdb/mxfold2_rnasdb_split_${SPLIT_ID}_train.pq
# https://wandb.ai/psi-lab/rna-sdb-mxfold2-train/runs/xpufl5zc


mamba activate mxfold2-train
cd /mnt/dg_shared_truenas/for_alice/work/mxfold2
export SPLIT_ID=5
WANDB_API_KEY=fc31024445b1bd60765be0295fb8f4a8ca0b389e  CUDA_VISIBLE_DEVICES=4 mxfold2 train --model MixC --param wkdir/rna_sdb/split_${SPLIT_ID}/model.pth --save-config wkdir/rna_sdb/split_${SPLIT_ID}/model.conf --gpu 0 --log-dir wkdir/rna_sdb/split_${SPLIT_ID}/  --epochs 1  /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/rna_sdb/mxfold2_rnasdb_split_${SPLIT_ID}_train.pq
# https://wandb.ai/psi-lab/rna-sdb-mxfold2-train/runs/42vag06b



mamba activate mxfold2-train
cd /mnt/dg_shared_truenas/for_alice/work/mxfold2
export SPLIT_ID=6
WANDB_API_KEY=fc31024445b1bd60765be0295fb8f4a8ca0b389e  CUDA_VISIBLE_DEVICES=5 mxfold2 train --model MixC --param wkdir/rna_sdb/split_${SPLIT_ID}/model.pth --save-config wkdir/rna_sdb/split_${SPLIT_ID}/model.conf --gpu 0 --log-dir wkdir/rna_sdb/split_${SPLIT_ID}/  --epochs 1  /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/rna_sdb/mxfold2_rnasdb_split_${SPLIT_ID}_train.pq
# https://wandb.ai/psi-lab/rna-sdb-mxfold2-train/runs/03l4tvwk



mamba activate mxfold2-train
cd /mnt/dg_shared_truenas/for_alice/work/mxfold2
export SPLIT_ID=7
WANDB_API_KEY=fc31024445b1bd60765be0295fb8f4a8ca0b389e  CUDA_VISIBLE_DEVICES=6 mxfold2 train --model MixC --param wkdir/rna_sdb/split_${SPLIT_ID}/model.pth --save-config wkdir/rna_sdb/split_${SPLIT_ID}/model.conf --gpu 0 --log-dir wkdir/rna_sdb/split_${SPLIT_ID}/  --epochs 1  /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/rna_sdb/mxfold2_rnasdb_split_${SPLIT_ID}_train.pq
# https://wandb.ai/psi-lab/rna-sdb-mxfold2-train/runs/h7q1gklm

mamba activate mxfold2-train
cd /mnt/dg_shared_truenas/for_alice/work/mxfold2
export SPLIT_ID=8
WANDB_API_KEY=fc31024445b1bd60765be0295fb8f4a8ca0b389e  CUDA_VISIBLE_DEVICES=7 mxfold2 train --model MixC --param wkdir/rna_sdb/split_${SPLIT_ID}/model.pth --save-config wkdir/rna_sdb/split_${SPLIT_ID}/model.conf --gpu 0 --log-dir wkdir/rna_sdb/split_${SPLIT_ID}/  --epochs 1  /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/rna_sdb/mxfold2_rnasdb_split_${SPLIT_ID}_train.pq
# https://wandb.ai/psi-lab/rna-sdb-mxfold2-train/runs/06575ewa

```


Inference: check wandb model artifact from above runs, make sure to use the corresponding one for testing  

- split 1: `psi-lab/rna-sdb-mxfold2-train/mxfold2-model:v8`

- split 2: `psi-lab/rna-sdb-mxfold2-train/mxfold2-model:v4`

- split 3: `psi-lab/rna-sdb-mxfold2-train/mxfold2-model:v6`

- split 4: `psi-lab/rna-sdb-mxfold2-train/mxfold2-model:v1`

- split 5: `psi-lab/rna-sdb-mxfold2-train/mxfold2-model:v3`

- split 6: `psi-lab/rna-sdb-mxfold2-train/mxfold2-model:v5`

- split 7: `psi-lab/rna-sdb-mxfold2-train/mxfold2-model:v7`

- split 8: `psi-lab/rna-sdb-mxfold2-train/mxfold2-model:v2`


```bash
mamba activate mxfold2-train
cd /mnt/dg_shared_truenas/for_alice/work/mxfold2
export SPLIT_ID=1
export MODEL_VER=8
WANDB_API_KEY=fc31024445b1bd60765be0295fb8f4a8ca0b389e  CUDA_VISIBLE_DEVICES=0 mxfold2 predict --model MixC --param psi-lab/rna-sdb-mxfold2-train/mxfold2-model:v${MODEL_VER} --gpu 0 --bpp wkdir/rna_sdb/split_${SPLIT_ID}/prediction.pq /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/rna_sdb/mxfold2_archiveII_split_${SPLIT_ID}.pq
# https://wandb.ai/psi-lab/rna-sdb-mxfold2-predict/runs/83pe6nlk


mamba activate mxfold2-train
cd /mnt/dg_shared_truenas/for_alice/work/mxfold2
export SPLIT_ID=2
export MODEL_VER=4
WANDB_API_KEY=fc31024445b1bd60765be0295fb8f4a8ca0b389e  CUDA_VISIBLE_DEVICES=0 mxfold2 predict --model MixC --param psi-lab/rna-sdb-mxfold2-train/mxfold2-model:v${MODEL_VER} --gpu 0 --bpp wkdir/rna_sdb/split_${SPLIT_ID}/prediction.pq /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/rna_sdb/mxfold2_archiveII_split_${SPLIT_ID}.pq
# https://wandb.ai/psi-lab/rna-sdb-mxfold2-predict/runs/nz6s91g3



mamba activate mxfold2-train
cd /mnt/dg_shared_truenas/for_alice/work/mxfold2
export SPLIT_ID=3
export MODEL_VER=6
WANDB_API_KEY=fc31024445b1bd60765be0295fb8f4a8ca0b389e  CUDA_VISIBLE_DEVICES=0 mxfold2 predict --model MixC --param psi-lab/rna-sdb-mxfold2-train/mxfold2-model:v${MODEL_VER} --gpu 0 --bpp wkdir/rna_sdb/split_${SPLIT_ID}/prediction.pq /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/rna_sdb/mxfold2_archiveII_split_${SPLIT_ID}.pq
# https://wandb.ai/psi-lab/rna-sdb-mxfold2-predict/runs/l5t5mjfu


mamba activate mxfold2-train
cd /mnt/dg_shared_truenas/for_alice/work/mxfold2
export SPLIT_ID=4
export MODEL_VER=1
WANDB_API_KEY=fc31024445b1bd60765be0295fb8f4a8ca0b389e  CUDA_VISIBLE_DEVICES=0 mxfold2 predict --model MixC --param psi-lab/rna-sdb-mxfold2-train/mxfold2-model:v${MODEL_VER} --gpu 0 --bpp wkdir/rna_sdb/split_${SPLIT_ID}/prediction.pq /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/rna_sdb/mxfold2_archiveII_split_${SPLIT_ID}.pq
# https://wandb.ai/psi-lab/rna-sdb-mxfold2-predict/runs/u37i6o75


mamba activate mxfold2-train
cd /mnt/dg_shared_truenas/for_alice/work/mxfold2
export SPLIT_ID=5
export MODEL_VER=3
WANDB_API_KEY=fc31024445b1bd60765be0295fb8f4a8ca0b389e  CUDA_VISIBLE_DEVICES=0 mxfold2 predict --model MixC --param psi-lab/rna-sdb-mxfold2-train/mxfold2-model:v${MODEL_VER} --gpu 0 --bpp wkdir/rna_sdb/split_${SPLIT_ID}/prediction.pq /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/rna_sdb/mxfold2_archiveII_split_${SPLIT_ID}.pq
# https://wandb.ai/psi-lab/rna-sdb-mxfold2-predict/runs/ilojskck



mamba activate mxfold2-train
cd /mnt/dg_shared_truenas/for_alice/work/mxfold2
export SPLIT_ID=6
export MODEL_VER=5
WANDB_API_KEY=fc31024445b1bd60765be0295fb8f4a8ca0b389e  CUDA_VISIBLE_DEVICES=0 mxfold2 predict --model MixC --param psi-lab/rna-sdb-mxfold2-train/mxfold2-model:v${MODEL_VER} --gpu 0 --bpp wkdir/rna_sdb/split_${SPLIT_ID}/prediction.pq /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/rna_sdb/mxfold2_archiveII_split_${SPLIT_ID}.pq
# https://wandb.ai/psi-lab/rna-sdb-mxfold2-predict/runs/5hx55tks


mamba activate mxfold2-train
cd /mnt/dg_shared_truenas/for_alice/work/mxfold2
export SPLIT_ID=7
export MODEL_VER=7
WANDB_API_KEY=fc31024445b1bd60765be0295fb8f4a8ca0b389e  CUDA_VISIBLE_DEVICES=0 mxfold2 predict --model MixC --param psi-lab/rna-sdb-mxfold2-train/mxfold2-model:v${MODEL_VER} --gpu 0 --bpp wkdir/rna_sdb/split_${SPLIT_ID}/prediction.pq /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/rna_sdb/mxfold2_archiveII_split_${SPLIT_ID}.pq
# https://wandb.ai/psi-lab/rna-sdb-mxfold2-predict/runs/u2k6fr34


mamba activate mxfold2-train
cd /mnt/dg_shared_truenas/for_alice/work/mxfold2
export SPLIT_ID=8
export MODEL_VER=2
WANDB_API_KEY=fc31024445b1bd60765be0295fb8f4a8ca0b389e  CUDA_VISIBLE_DEVICES=0 mxfold2 predict --model MixC --param psi-lab/rna-sdb-mxfold2-train/mxfold2-model:v${MODEL_VER} --gpu 0 --bpp wkdir/rna_sdb/split_${SPLIT_ID}/prediction.pq /mnt/dg_shared_truenas/for_alice/work/rna_sdb/datasets/rna_sdb/mxfold2_archiveII_split_${SPLIT_ID}.pq
# https://wandb.ai/psi-lab/rna-sdb-mxfold2-predict/runs/p63057mw

```



send results to Andrew, annotate wandb artifact ID (from my project) for reproducibility:


- split_1: local: `wkdir/rna_sdb/split_1/prediction.pq`: wandb: `psi-lab/rna-sdb-mxfold2-predict/mxfold2-prediction:v7`

- split_2: local: `wkdir/rna_sdb/split_2/prediction.pq`: wandb: `psi-lab/rna-sdb-mxfold2-predict/mxfold2-prediction:v4`

- split_3: local: `wkdir/rna_sdb/split_3/prediction.pq`: wandb: `psi-lab/rna-sdb-mxfold2-predict/mxfold2-prediction:v1`

- split_4: local: `wkdir/rna_sdb/split_4/prediction.pq`: wandb: `psi-lab/rna-sdb-mxfold2-predict/mxfold2-prediction:v2`

- split_5: local: `wkdir/rna_sdb/split_5/prediction.pq`: wandb: `psi-lab/rna-sdb-mxfold2-predict/mxfold2-prediction:v6`

- split_6: local: `wkdir/rna_sdb/split_6/prediction.pq`: wandb: `psi-lab/rna-sdb-mxfold2-predict/mxfold2-prediction:v5`

- split_7: local: `wkdir/rna_sdb/split_7/prediction.pq`: wandb: `psi-lab/rna-sdb-mxfold2-predict/mxfold2-prediction:v3`

- split_8: local: `wkdir/rna_sdb/split_8/prediction.pq`: wandb: `psi-lab/rna-sdb-mxfold2-predict/mxfold2-prediction:v8`


Copy and rename the files and put in one folder for Andrew:

```bash
export RNASDB_PRED_DIR="/mnt/dg_shared_truenas/for_alice/data/rna_sdb/2024_05_27"
mkdir -p $RNASDB_PRED_DIR

cp wkdir/rna_sdb/split_1/prediction.pq ${RNASDB_PRED_DIR}/prediction_split_1.pq
cp wkdir/rna_sdb/split_2/prediction.pq ${RNASDB_PRED_DIR}/prediction_split_2.pq
cp wkdir/rna_sdb/split_3/prediction.pq ${RNASDB_PRED_DIR}/prediction_split_3.pq
cp wkdir/rna_sdb/split_4/prediction.pq ${RNASDB_PRED_DIR}/prediction_split_4.pq
cp wkdir/rna_sdb/split_5/prediction.pq ${RNASDB_PRED_DIR}/prediction_split_5.pq
cp wkdir/rna_sdb/split_6/prediction.pq ${RNASDB_PRED_DIR}/prediction_split_6.pq
cp wkdir/rna_sdb/split_7/prediction.pq ${RNASDB_PRED_DIR}/prediction_split_7.pq
cp wkdir/rna_sdb/split_8/prediction.pq ${RNASDB_PRED_DIR}/prediction_split_8.pq
```



### Instruction on training and inference


#### Installation


```bash
mamba create -n mxfold2-train python=3.9 ipython

mamba activate mxfold2-train

mamba install numpy pandas pyarrow pybind11 pytorch==1.13.1 torchvision pytorch-cuda=11.7 tqdm wandb wheel -c pytorch -c nvidia

pip install .
# for dev: pip install --editable .
```

Once installed, command `mxfold2` can be evoked from anywhere.

:exclamation:  Make sure to install `pyarrow` not `fastparquet`, so that we can load Andrew's `pq` dataset. 


:exclamation: Note that if working off GPU server and using `truenas` the dev-mode (`pip install --editable .`) doesn't seem to work. Instead do `pip install .`.


#### Dataset

Prepare your dataset as dataframe and save in `parquet` format. The following 3 columns are required:

- seq_id

- seq

- pair_indices


#### Training


```bash
CUDA_VISIBLE_DEVICES=0 mxfold2 train --model MixC --param {local-file-path-to-export-model-params} --save-config {local-file-path-to-export-model-config} --gpu 0 --log-dir {local-dir-to-export-log-and-checkpoint}  --epochs {num-epoch}  {local-file-path-to-dataset.pq}
```


@andrew: I believe you can also omit `--param` and `--save-config` since I've modified the script to upload the pytorch checkpoint that contains everything.

@andrew: the script upload model as wandb artifact, please make a note of its ID. It is needed in inference.


@andrew: `train.py` is hard-coded to log to my wandb, you'll need to change that.


#### Inference


```bash
CUDA_VISIBLE_DEVICES=0 mxfold2 predict --model MixC --param {wandb-model-id} --gpu 0 --bpp {local-file-path-to-export-prediction.pq} {local-file-path-to-dataset.pq}
```

@andrew: prediction parquet file is also uploaded as wandb artifact.

@andrew: `predict.py` is hard-coded to log to my wandb, you'll need to change that.
