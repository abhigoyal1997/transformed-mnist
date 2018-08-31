# CNNs on Transformed MNIST

Analysis of different CNN models on a randomly scaled and translated MNIST dataset using a multi-label setup (for generalisation in case of multi-digit).

## Some Results

These models have been trained on 10000 images from official training split of MNIST after random scaling and translation using a Multi-Label-Soft-Margin loss. The results are reported as average F1-score of prediction on the official 10000 test images from MNIST after random scale and translation.

1. 2 Convolution layers followed by 3 Dense layers: ~0.732
2. A model similar to AG-CNN:
    * Global branch (the model mentioned above): ~0.732
    * Local branch using localized images: ~0.938
    * Fused Global and Local branch: ~0.957

## References

1. AG-CNN: <a href="https://arxiv.org/abs/1801.09927">Diagnose like a radiologist: Attention guided convolutional neural network for thorax disease classification</a>
2. <a href="http://yann.lecun.com/exdb/mnist/">MNIST Dataset</a>
