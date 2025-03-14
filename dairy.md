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