# 4L13NHUN73RS
Satellite Data Imagery With Variational Autoencoders

Code is based on this project: https://gist.github.com/danijar/1cb4d81fed37fd06ef60d08c1181f557


#CIFAR10 Branch

This branch is dedicated for making the code compatible with the CIFAR10 dataset.
Currently only using data_batch_1 to grab images, and only selecting cat images


*Errors*: Code currently breaks when running, issue seems to be on line 154 when attempting to create the elbo
Relevant Error:
`tensorflow.python.framework.errors_impl.InvalidArgumentError: Incompatible shapes: [1016,32] vs. [1016]`
