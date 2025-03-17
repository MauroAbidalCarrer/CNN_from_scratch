11/03/2024: Succesfully trained small cnn on 4 cifar10 samples by having a very low learning rate and a lot (~1k) epochs.  
12/03/2024: better conv layer (looked at fft but ended up simply switching from einsum to tensordot)  
13/03/2024: 
  - max pool
  - Almost successfully trained on 7 samples cifar10.  
    Reached accuracy of 100% and then it seems like the gradients explode  

13/03/2024: 
- Added layers weights means as metrics in traing stats df.  
  They seem to confirm that the gradients AND the weights increasing and then exploding.  
- batch norm  

17/03/2025:  
- Watched these videos about softmax:  
  - https://www.youtube.com/watch?v=ytbYRIN0N4g&ab_channel=ElliotWaite  
  - https://www.youtube.com/watch?v=p-6wUOXaVqs&ab_channel=MutualInformation  
- Learned that the aim of Batch norm layer is not to mean center and "unit variate" the inputs.  
  It's also an affine transformation after that.
  Its implementation is a bit more complicated than I thought it would be...
  Might switch to implementing the adam optimzer instead...