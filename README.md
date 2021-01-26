# &nbsp;Tahoe: Tree Structure-Aware High Performance Inference Engine for Decision Tree Ensemble on GPU
# (this repo will be updated in two days)

Using decision trees for inference on GPU is challenging, because of irregular memory access patterns and imbalance workloads across threads. Tahoe, an tree structure-aware high performance inference engine for decision tree ensemble, rearranges tree nodes to enable effective and coalesced memory accesses, and also rearranges trees, such that trees with similarity structure are efficiently grouped together in memory and assigned to threads in a balanced way. Tahoe introduces a set of inference algorithms, each of which uses shared memory differently and has different implications on reduction overhead. It also introduces performance models to guide the selection of inference algorithms for arbitrary forest and data set with negligible overhead.

Tahoe enables high-performance GPU inference for decision tree ensemble which can be used in high performance computing, advertising systems, medical diagnosis, financial fraud detection, etc.

As an example, the following C++ code loads forest and data from files and computes the prediction on GPU:
```C++
# Setup input forest, dataset, and algorithm, and construct the interface of Tahoe framework
BaseTahoeTest* pTest = new BaseTahoeTest(argv[1], argv[2], algorithm);
# Launch the inference of tree traversal
pTest->SetUp();
# Free resources
pTest->Free();
```

## Download trained forest and inference data
```
# Take the SVHN data set as an example.
# Download trained forest
https://drive.google.com/file/d/1cvKjrYR9PHsvNUrZkDWM4u-CMuWVvSey/view?usp=sharing
# Download data set
https://drive.google.com/file/d/1qb1Jr5nAB1pXWGnpP8_yj3KZWAQQu3MO/view?usp=sharing
```

## Build/Install from Source
```
git clone https://github.com/zhen-xie/Tahoe.git
cd Tahoe;
make Tahoe;
```

## Run an example
```
# Move two downloaded files to this folder
# Run Tahoe with the corresponding paths. The first parameter is the model path, and the second parameter is the data path.
./Tahoe [PATH_MODEL] [PATH_DATASET]
```

## Output
```
[zxie10@gpu050 Tahoe]$ ./Tahoe /home/zxie10/Project1/model_SUSY.txt /home/zxie10/Project1/data_SUSY.txt;
Performance model choose #1 strategy.
/home/zxie10/Project1/model_SUSY.txt
Load forest
Load data
Predict on CPU and get standard results...
Test on GPU...
time_dense is 0.990169 us
results are correct
Using strategy 0
time_dense is 1.082617 us
results are correct
Using strategy 0
time_dense is 3.838786 us
results are correct
Using strategy 1
time_dense is 0.120020 us
results are correct
Using strategy 3
time_dense is 0.212903 us
results are correct
Enumerate al strategies and choose #1 strategy.
Performance model predicts correctly
Tahoe brings 8.25x speedup.
```
