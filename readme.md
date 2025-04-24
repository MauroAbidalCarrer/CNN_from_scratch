# CNN from scratch using only Numpy

This is a toy project for me learn the inner workings of:
- computer vision model architectures
- Optimizers

## Source code:
This is effectively a very small (500 lines)Deep learning library with the following features:
Layers:
    - Linear
    - Relu
    - Softmax
    - Convolutional
    - BatchNorm
    - MaxPool

Optimizers:
    - Stochiastic Gradient Descent (SGD)
    - SGD with momentum
    - AdaGrad
    - Adam

> Note: 
> - The first three optimizers have been removed as Adam ended up being the only one I would use.
> - The Adam optimizer has live training stats using plotly which I think is pretty cool.

## Results:
The main focus of this project was to fit on the **cifar10** dataset.  
Unfortunately I only managed to fit up to **60% accuracy on the validation** set.  
Despite this unsatisfactory result, I decided to move on and start programming on pytorch.  
This is because I was spending more time waiting for the result of my implementations than actually trying new implementation of features to get a better validation score.  
