# PolicyNet

This repository is intended as a complement to the article: "", and contains the code related to the shown results.

The data used come from the [DDD20](https://sites.google.com/view/davis-driving-dataset-2020/home) dataset. The two extracted recordings are split in this repository, because of the limits on the file size. In order to reconstruct the files, please run in your terminal:

```
cat train* > rec1501614399_export.hdf5
```

```
cat test* > rec1501612590_export.hdf5
```

This repository contains Python scripts and IPython Notebooks, organised as follows:

- The folders `./cnn300` and `cnnlstm300` contain the scripts for performing 300 epochs of training on the identified state of the art models.

- policyNet.py is the script in which the model is defined, trained and evaluated. After testing, the script saves the predicted values to a file, for any need of further inspection.

- The Notebook `pt_to_csv.ipynb` lets the user convert the files containing the predictions obtained by running the scripts in the repository by running the related cell.
