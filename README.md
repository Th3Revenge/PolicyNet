# PolicyNet

This repository is intended as a complement to the thesis: "Design and Implementation of an Online Learning Model targeting a Resource-Constrained Parallel-Computing environment", and contains the code related to the shown results.

The data used come from the [DDD20](https://sites.google.com/view/davis-driving-dataset-2020/home) dataset. The two extracted recordings are split in this repository, because of the limits on the file size. In order to reconstruct the files, please run in your terminal:

```
cat train* > rec1501614399_export.hdf5
```

```
cat test* > rec1501612590_export.hdf5
```

This repository contains a combination of Python scripts and IPython Notebooks, organised as follows:

- `./vit/` folder contains the scripts for performing 300 epochs of training of the network proposed in the thesis, with images representations obtained by using ViT. Included scripts are: `represent.py`, which generates image representations; `train.py`, which performs the training step after that images have been represented; `test.py` in charge of evaluating the performance of the model on the test set.

- `./swin/` folder contains script doing the exact same thing as before, but using Swin-Transformer for obtaining image representations

- `./notebooks/ViT-vs-Swin-Transformer.ipynb` performs training, evaluation, and performance comparison for the two previous cases, but for one epoch of training, simulating the Online Learning logic described in the thesis.

- The folders `./cnn300` and `cnnlstm300` contain the scripts for performing 300 epochs of trainnig on the identified state of the art models.

- The results after one epoch of training on the state of the art models are obtained and evaluated into `./notebooks/sota_comparison.ipynb`.
