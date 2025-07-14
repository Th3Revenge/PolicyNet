# PolicyNet

This repository is intended as a complement to the thesis: "Design and Implementation of an Online Learning Model targeting a Resource-Constrained Parallel-Computing environment", and contains the code related to the shown results.

The data used come from the [DDD20](https://sites.google.com/view/davis-driving-dataset-2020/home) dataset. The two extracted recordings are split in this repository, because of the limits on the file size. In order to reconstruct the files, please run in your terminal:

```
cat train* > rec1501614399_export.hdf5
```

```
cat test* > rec1501612590_export.hdf5
```
