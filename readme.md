# Implementation of SphereFace: Deep Hypersphere Embedding for Face Recognition

SphereFace uses A-Softmax Loss, trains and evaluates on LFW dataset.
Implementation in Python 3.8 and PyTorch 1.7.1.

Place LFW data in `./data` following

    .
    ├── lfw  # image folder
    ├── pairsDevTest.txt
    ├── pairsDevTrain.txt

To run the code, use `python main.py`. Default using GPU 0.

This implementation gets 65.80% on LFW dataset.