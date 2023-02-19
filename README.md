# AdderNet-on-tensorflow
a implementation of L1 Convolution and Dense layer with tensorflow 2 (Adder Net).

this project is still under development.

# Requirement
* tensorflow 2
* numpy

# Work log

At first, the training accuracy was 96% and the testing accuracy was 95.6% on the MNIST dataset.

[2023-02-03] add full-precision gradient, the training accuracy is 97.4% and the testing accuracy is 96.9% on the MNIST dataset.

[2023-02-04] add gradient clipping, the training accuracy is 97.1% and the testing accuracy is 96.4% on the MNIST dataset.

[2023-02-16] add cosine lr, the training accuracy is 99.97% and the testing accuracy is 98.87% on the MNIST dataset.

[2023-02-18] add adaptive gradient scaling, change the optimizer from adam to NAG and increase the epoches to 50, the training accuracy is 99.99% and the testing accuracy is 98.94% on the MNIST dataset.
