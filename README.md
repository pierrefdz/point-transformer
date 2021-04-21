[Paper](<https://arxiv.org/abs/2012.09164>) | [Colab](<https://colab.research.google.com/drive/18-r47vgJSdtQkfIzKkadfpQtEpEf0Y9Q?usp=sharing>)  


# Unofficial Implementation of Point Transformer for classification on ModelNet40

*Point Transformer* is a paper by Hengshuang Zhao et al. (2020). The project was done in the scope of the course NPM3D of Master MVA 2021. It consisted in understanding the paper and testing one part of the paper, here the classification on ModelNet40.

I also provide a notebook where all the pipeline is already implemented.

## Dataset

The data can be downloaded [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip). It is a pre-processed version of the ModelNet40 CAD models that was created for PointNet++ (see [PointNet++]{http://stanford.edu/~rqi/pointnet2/}).
It must then be unzipped and saved in `data/modelnet40_normal_resampled`. 

Loading the dataset takes a long time the first time, since it pre-computes the samplings and saves it in the files `modelnet40_test_1024pts_fps` and `modelnet40_train_1024pts_fps`. Once pre-processed it allows the computations to be much faster for the rest of the pipeline.

## Run 

To run a training procedure and test a model you should run `train.py` (and change parameters if you want to). In the current configuration, you should obtain around 91.5% accuracy on the test split.

Each .py file can be unit-tested (by directly running them)
