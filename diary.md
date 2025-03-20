11/03/2024: Succesfully trained small cnn on 4 cifar10 samples by having a very low learning rate and a lot (~1k) epochs.  
12/03/2024: better conv layer (looked at fft but ended up simply switching from einsum to tensordot)  
13/03/2024: 
  - max pool
  - Almost successfully trained on 7 samples cifar10.  
    Reached accuracy of 100% and then it seems like the gradients explode  
- Added layers weights means as metrics in traing stats df.  
  They seem to confirm that the gradients AND the weights increasing and then exploding.  

17/03/2025:  
- Watched these videos about softmax:  
  - https://www.youtube.com/watch?v=ytbYRIN0N4g&ab_channel=ElliotWaite  
  - https://www.youtube.com/watch?v=p-6wUOXaVqs&ab_channel=MutualInformation  
- Learned that the aim of Batch norm layer is not to mean center and "unit variate" the inputs.  
  It's also an affine transformation after that.
  Its implementation is a bit more complicated than I thought it would be...
  Might switch to implementing the adam optimzer instead...
- Reread the NNFS chapters on optimizers
- Started implementing SGD and SGD_with_decay and refactoring layers to work with it.

18/03/2025:
- Tested refacto on mnist notebook, works fine :thumbsup:.
- Tested SGD_with_decay on 10 samples of cifar10, it converges int [400, 700] epochs and then diverges.
  I can't get it to stay at a minimum loss/accuracy...
  I will try to implement Addam to see if I can get it to converge and stay at a satisfactory loss minima/accuracy maximum.
  Actually I am first going to try to fit an nn with:
    - [Flatten, Conv, Softmax] This is to make sure that the issues I am encountering in the cnn's training are not a direct cause to some bad Conv implementation
      It's not working, it just stagnates or has big unexpected/e=unexplicable jumps in loss but never converges
    - [Flatten, Linear, Sigmoid] To see if the issues encountered in the training of the single conv layer cnn are caused by the Conv layer since a conv layer of the same shape as its input is (if I am not mitaken) the same as a Linear layer.
      And........... it's also not working -_-, it exhibits the same weird training patterns.
      I am assuming this is due to something else then, maybe the loss?
      By looking into it, it turns out that the gradients are so absurdly small that the substraction (that I have perfomed in a notebook just to be sure) gives the same param.
      *I also noted that the single Linear layer nn is ~40x faster than the single Conv layer nn so I will definetly look into (yet another) better Conv implementation.*

19/03/2025:
- Ok so it turns out that the learning rate was just too high, I set it to 0.03 and it works just fine with the [Flatten, Linear, Softmax] nn.
  However, it does not converge with the conv layer which, again is weird since it should be exactly the same thing...
  The [Flatten, Conv, Softmax] nn required a 0.0005 starting lr and a 0.005 lr decay where as the [Flatten, Linear, Softmax] nn didn't even need lr decay...
  I believe this is worth investing, let's see if this is caused by my implementation or an actual/real property of Conv layers.
  Ok, I checked with uncleGPT and it suggest (among other things thatI don't believe are worth looking into) that it might be due to the way gradients are computed.
  AND infact the gradients wrt kernels are computed as the full_convolve of the input and output I assuming that this is the reason for that diff and will move on to not get caught in, yet another, rabbit hole.
- Still, I can't get the [2x[Convolutional, Relu, MaxPool], Flatten, Linear, Relu, Linear, Softmax] nn to converge.
  I tried to tweak the starting lr and lr decay for 1-2 hours and it never got above 50% accuracy within ~400 / 1000 epochs.
  Currently it takes about 35 seconds to train the nn for 400 epcohs, lets try to improve this.
- I implemented a function that is ~30% faster than the tensordot current implementation with the 6 samples in the notebook.

20/03/2025:
- Testing the new cross corr implementation on 300 and 3000 samples, it is actually slower in both cases:
  at 300 samples: `@` and `tensordot` take the same amount of time but computing and flattening the views takes more time (obviously) than just computing the views.
  at 3000 samples: `@` takes more time than `tensordot`.
  So I will stay with the tensordot implementation.
- I will now get back to fitting the [2x[Convolutional, Relu, MaxPool], Flatten, Linear, Relu, Linear, Softmax] nn on htethe 10 samples cifar10 subset.
  Before starting to implement Adam I tried the following:
    - training a new nn 4 times, 
    - checking the iteration where the convergence point happens.
    - averaging the convergence iteration point
    - getting the learning cumsum up to this point
    - declaring a desired learning rate at this point as the learning rate at this convergence point dvided by an arbitrary denominator
    - computing the starting lr and lr decay based on the desired lr cumsum and desireed lr at the convergence point.
    I tried it multiple times but it never worked...
- I looked at [this notebook](https://www.kaggle.com/code/valentynsichkar/convolutional-neural-network-from-scratch-cifar10#Creating-Convolutional-Neural-Network-Model).
  Interesingly enough it uses only one 32 filters conv layer and two FC layers 
- Ok, I will start to implement Adam once and for all...
  Implemented SGD_with_momentum, it works, the mnist MLP converges faster with it... but it doesn't help me fit the nn to the cifar10 subset .
  I do not a different loss curve in the cifar10 subset nn training after the convergence point but nothing seems to change before...
  I will now start implementing AdaGrad.