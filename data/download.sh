#!/bin/bash


for image in train-images-idx3 train-labels-idx1 t10k-images-idx3 t10k-labels-idx1
do
  wget http://yann.lecun.com/exdb/mnist/$image-ubyte.gz
  gunzip $image-ubyte.gz
done

