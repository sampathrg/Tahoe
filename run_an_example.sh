#!/bin/sh

if ! type "nvcc" > /dev/null; then
  # install foobar here
  echo "No NVCC (NVIDIA CUDA Compiler) detected"
  echo "Using this command ''sudo apt install nvidia-cuda-toolkit'' to install it. Or see https://developer.nvidia.com/Cuda-downloads"
fi


#download trained model for SVHN data set
echo "Downing trained model for SVHN data set..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Z04IvVIIZdjUGZGIf4CpUQ2Ll06JVVll' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Z04IvVIIZdjUGZGIf4CpUQ2Ll06JVVll" -O ./models/model_SVHN.txt && rm -rf /tmp/cookies.txt

#download SVHN data set
echo "Downing SVHN data set..."
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1sqnuc8O9c58qphMHYIaNZSiHdDKV0AMF' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1sqnuc8O9c58qphMHYIaNZSiHdDKV0AMF" -O ./data/data_SVHN.txt && rm -rf /tmp/cookies.txt

#compile the Tahoe
make;

#run an example
./Tahoe models/model_SVHN.txt data/data_SVHN.txt;
