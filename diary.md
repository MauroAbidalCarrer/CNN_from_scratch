11/03/2024: Succesfully trained small cnn on 4 cifar10 samples by having a very low learning rate and a lot (~1k) epochs.  
12/03/2024: better conv layer (looked at fft but ended up simply switching from einsum to tensordot)  
13/03/2024: 
  - max pool
  - Almost successfully trained on 7 samples cifar10.  
    Reached accuracy of 100% and then it seems like the gradients explode  
- Added layers weights means as metrics in traing stats df.  
  They seem to confirm that the gradients AND the weights are increasing and then exploding.  

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
  I implemented SGD_with_momentum and RMSprop.

21/03/2025:
- Implemented Adam optimizer.
  It improved the mnist score training accuracy from 0.93to 0.98!
  Damn it fitted the [2x[Convolutional, Relu, MaxPool], Flatten, Linear, Relu, Linear, Softmax] nn on the 10 samples cifar10 subset first try in 37 epochs wtf!!
- Watched a big portion of this [video](https://www.youtube.com/watch?v=l3O2J3LMxqI&t=3248s&ab_channel=MachineLearningStreetTalk)

22/03/2025:
- Started watching this [video](https://www.youtube.com/watch?v=7Q2JhZxNPow&ab_channel=SimonsInstitute)
- Getting ~60% on 100 samples cifar10 subset with the same nn and Adam
- Reached  80% by tweaking the Adam hyperparameters (mostly starting_lr and lr_decay).
    starting_lr=0.01,
    lr_decay=0.0001,
    momentum_weight=0.99,
    ada_grad_weight=0.99,
    epochs=200

23/03/2025:
- Watched almost all of this [video](https://www.youtube.com/watch?v=78vq6kgsTa8&t=663s&ab_channel=InstituteforPure%26AppliedMathematics%28IPAM%29).

24/03/205:
- Turns out that the nn doesn't always fits to ~80% on the 100 samples subset, most of the times it doesn't...
- Looking for ressources that explain **why** a network is not fitting.
  Find this [PDF](https://misraturp.gumroad.com/l/fdl), looks interesting.
- Going to try gradient clipping to see if it improves the training.
- Looked at plotly FigureWidgets to stream the training data to the figure. I will defenetly be using that instead of rich.progress.track + base plotly Figure/
- Reached 97% with the same hyper params as the 22/03/2025, interestingly enough, the abs gradient mean (almost) never went above 2. 
- Also notted that the undefitting nns seem to have  damped sin wave like loss curve where as the fitting ones seem to have a 1 - sigmoid like loss curve.
  This, hopefully, means that there is only one problem to solve, hoperfully...
- Started to refacto optimizers module to be smaller, more modular and use FigureWidget instead of rich.progress.track and then px.scatter
- FigureWdiget works fine BUT, it takes 2x more time to train an nn with it AND it doesn't render when you reopen the notebook so I might switch back to track + Figure.
  Though it is usefull to see the metrics live so idk I will leave it as is for now.

25/03/2025:
- Find out about dataclasses.dataclass and dataclasses.field, using them for the new Adam class.
- Read the 6.6 chapter of UnderstandingDeepLearning.
  Reading it, I learned that the point of having batches of the dataset at each step is to have a different gradients for the same model and avoid getting stuck in local minimas.
  > For nonlinear functions, the loss function may have both local minima (where gradi-  
  > ent descent gets trapped) and saddle points (where gradient descent may appear to have  
  > converged but has not). Stochastic gradient descent helps mitigate these problems.1 At  
  > each iteration, we use a different random subset of the data (a batch) to compute the  
  > gradient. This adds noise to the process and helps prevent the algorithm from getting  
  > trapped in a sub-optimal region of parameter space.  
  I tried to reduce the batch_size to 10 but it didn't seem to change anyhting...
  Maybe I should try to permutate the samples at each epoch to further randomize and the batch gradients(and maybe decrease momentum?)
  I also notted that the underfitting CNNs losses plateau at the same value(~3.27) which, I assume, is related to the distribution of labels in the 100 (first) samples.
  I will try to sample an equal amount of labels and see if I can more relaiably get a fitted model.

26/03/2025:
  - Made even x_train/y_train of 100 samples cifar10 subset doesn't seem to improve anything...
  - Implemented `nn_params_stats` and `activations_stats` metric functions to get their mean, std, l1 and l2.
  - I think the best way to understand why some models are underfitting and some others fitting, would be to get ~5 training stats (including activations, gradients AND params).
    And then try to find some meaningfull property that explains why some are fitting and some not. 
  - I will first implement a way to read/write a network to a json

27/03/2025:
  - Actually I first wrote a while loop that breaks once a fitting model has been found.
    The first time I ran it it got a 50% accuracy model but I messed up a line and the best model didn't get saved...
  - I re ran the loop a few times, trained abut 100 or so models, and obviou-fuckingly now no model is fitting (-_-).
  - Maybe I should do the study with models trained on the 10 smaples cifra10 subset instead of the 100 samples cifra10 subset as I get way more fitted models.
    Or speed up the training, if I could find a 10x improvement it would be enough to reliably find a fitting model in ~10mins... I think.
  - I looked at the AlexNet architecutre and it reminded that the number of kernels should increase as the are further in the model.
    My current architecture is:
    ```python
    [
        Convolutional((10, 5, 5, 3)),
        Relu(),
        MaxPool((2, 2)),
        Convolutional((10, 3, 3, 10)),
        Relu(),
        MaxPool((2, 2)),
        Flatten(),
        Linear(360, 64),
        Relu(),
        Linear(64, y.shape[1]),
        Softmax(),
    ]
    ```
    There are as many kernels as there are classes in the second conv layer wich might be problematic actually...
    I changed it to:
    ```python
    [
        Convolutional((20, 5, 5, 3)),
        Relu(),
        MaxPool((2, 2)),
        Convolutional((32, 3, 3, 20)),
        Relu(),
        MaxPool((2, 2)),
        Flatten(),
        Linear(1152, 64),
        Relu(),
        Linear(64, y.shape[1]),
        Softmax(),
    ]
    ```
    Let's see if that fixes it, it certainly takes forwever to train a model for 100 epochs...  
    6 minutes in and only 3 models trained, best accuracy is 12% -_-.  
    22 minutes in and the best accuracy is 14%...  
    While it is annoying that I can't find a fitting model anymore, the while loop is at least an effective way to test an idea.  
    After an hour and ~18 more trained models no improvements so I will switch back to the previous architecture.  
  - Let's try the btach norm layer.   
    Though, looking at the AlexNet architecture again,    
    I made chatGPT implement it, I was suprised to see that the mvg average/std, gamma and beta ar of shape [1, 1, 1, channels] and not [1, width, height, channels].   
    ChatGPT says that it's to preserve the signal that lies inbetween the values(he didn't phrase it that nicely, I did).   
    I guess that makes sense to me but following that logic it would also mean that there is a singal that lies in between the channels right?    
    Anyways, the results are unanimous, BatchNorm is amazing. Almost all the models fit.    
    Damn, I literally almost implemented it 10 days ago...   
  - I'm too tired to keep coding (or too lazzy idk...) so I watched this ["why batchnorm works" deeplearning.ai video](https://www.youtube.com/watch?v=nUUqwaxLnWs&ab_channel=DeepLearningAI).  
    Reminded me that I should look into deeplearning.ai.  
    It was great, made me understand why it has a small regularizatin but it didn't explain its effect on gradients.

28/03/2025:
  - I started implementing BatchNorm by myself because I didn't really like the xhatGPT implementation, even though it works jsut fine.  
    I wanted to implement the normalization as `(inputs - mean) / (std - epsilon)` instead of `(inputs - mean) / sqrt(variance + epsilon)`.  
    The latter was in the original paper of the batchnorm layer so I thought that there was probably a reason for the presence of the sqrt.
    I asked chatGPT if these were equivalent and it turns out that it isn't.  
    The latter is better because its output for small variance inputs is properly mean centred (the mean value is the same but numerical instability causes very small numbers to not get mean centerd output).  
    And the outputs variance is also closer to one for the second method.  
    I'm not really sure it would impact that much the training but let's not take the risk just to remove one sqrt call, shall we?  
  - Updated the repo with all the modifications and rewrote BatchNorm to my liking, I will try to perform batchnorm over other axis combinations than "just" (0, 1, 2) and apply it on the fc layers (and input layer?).  

29/03/2025:
  - The network works fits properly on 1000 samples cifar10 subset, though it takes forever to train (~10 mins).
  - Trying on 5k samples, it fits up to ~50% accuracy and then the accuracy (and loss) plateaus.
  - I might need to switch implementation because to speed things up, it's a shame there isn't a simple numpy on GPU alternative, I looked into numba and it seems to have a lot of caveats...
  - Looking at the cifar10 from scratch kaggle notebook I (re)saw that the model had only one (bigger, 32x7x7x3 wher I have 10x5x5x3) conv layer.  
    And a hidden FC layer of output size 300.
    I.e a shorter wider network.
    I tried to mimicking this by removing the second conv layer and set the ouput size of the hidden FC layer to 128.
    It works!
    I notted that the output size of the first FC hidden layer (64) often turned out to be a bottleneck.

31/03/2025:
  - Maybe I should switch from `tensordot` to `scipy.signal.convolve` which under the hood can use `fft`.
  - The kaggle cifar10 with numpy kaggle notebook uses a beta_1 of 0.8 (which corresponds to the `momentum_weight` in my Adam optimazation).
    I tried the same value for my training and it did speed things up: the model plateaus to 50% earlier.  
    It's not a real improvement but at least it should help me find a solution faster as it will take me less time to check if a solution works.  
  - I added back the second conv layer and it seems to work a lot better, reaching 72% accuracy at epoch 50.
  - I increased the number of filters in the second conv layer from 10 to 32 but it didn't seem to really increase the perfs. 
  - Reached 85% accuracy after 100 epochs... and 30ins.
  - I will try to decrease tthe number of filters in the second conv layer to 20 and increase the size of the filters of the first from 5x5 to 7x7.
  - Even with this fairly high accuracy on this training data subset, I only get 33% accuracy on the test data.
    That is the same score as the nn trained on the 1k samples subset.
    That being said the x_test is 10k samples so it's not that suprising.

01/04/2025:
  - Looked into a implementation of [cnn on cifar10 with cuda](https://www.kaggle.com/code/alincijov/cifar-10-numba-cpu-cnn-from-scratch) but it didn't seem any faster.
    > Note: I had to fix this line `for i in range(len(df1)):` into `for i in range(len(df1[0])):` (and turn the internet of the notebook on but this is not a fix).  

    So maybe I shouldn't work with numba?
    I found another [cnn with cuda](https://github.com/WHDY/mnist_cnn_numba_cuda/tree/master) let's look into that.  
    The (numpy) kaggle notebook used only 10 epochs so I should be able to get away with numpy and my desktop.   
    Maybe I should look into the weight initialization and try to replicate all the settings to see there is an improvement I can find and a lesson to learn from it.  
    Actually it has the same weight initialization as me...
    Let's try to set the hidden fc output size to 300.

02/04/2025:
- Actually, now that I think about it the full dataset is 50k i.e 10x the size of my 5k subset so it's not shocking that the numpy kaggle notebook needs 10x less epochs.
  - I'm going to look into CuPy, it also seems interesting
  - I tried a bunch of chatGPT written numba versions of my valid correlate but none of them were faster(sad)...
  - I tested the numpy cnn kaggle notebook and it is super slow.  
  - So back on the cupy option I would need to setup a remote session... let's try on kaggle.  
    Looking at this [cupy CNN repo](https://github.com/AlaaAnani/CNN-from-Scratch-Cupy), I stumble upon the lealkyReLu, I wander if this could speed up my training.  
    Letme ask daddyGPT and uncle Google.  
    Seems like no.... but I might try it anyway later.  
  - Saw that cupy does not provide `sliding_window_view`, but it does provide `as_strided` which I believe can be used for the same goal.   
    In fact, looking at the source of `sliding_window_view` we can see that it is built on top of `as_strided`:thumbsup:.  
  - As expected, setting up the kaggle notebook is an absolute pain in the ass...  
    Gonna try lightning AI.  
    Tried it, setup was pretty straight forward which is really cool.    
    However, when I tried this chatGPT written code snippet:  
    ```python
    import numpy as np
    import cupy as cp
    import time

    # Generate random inputs
    views_np = np.random.rand(10000, 26, 26, 3, 7, 7)
    k_np = np.random.rand(32, 7, 7, 3)

    # NumPy tensordot benchmark
    start = time.time()
    np_result = np.tensordot(views_np, k_np, axes=([3, 4, 5], [3, 1, 2]))
    np_time = time.time() - start
    print(f"NumPy tensordot Time: {np_time:.4f}s")

    # Transfer to GPU (CuPy)
    views_cp = cp.asarray(views_np)
    k_cp = cp.asarray(k_np)

    # CuPy tensordot benchmark
    start = time.time()
    cp_result = cp.tensordot(views_cp, k_cp, axes=([3, 4, 5], [3, 1, 2]))
    cp_time = time.time() - start
    print(f"CuPy tensordot Time: {cp_time:.4f}s")

    # Transfer back to CPU for comparison
    cp_result_np = cp.asnumpy(cp_result)

    # Validate correctness
    assert np.allclose(np_result, cp_result_np, atol=1e-5)
    print("Results match!")
    ```
    The time for numpy and cupy were equal (1.3s).  
    That suprised me in two ways:
    1.  Why isn't cupy faster?!!!!!
    1.  1.3s seems pretty fast for numpy letme compare this to my computer.  
    Ran it on my computer and it's infact a lot slower: 5s.  
    So even if I don't use the lightning GPU I would still get a 5x speed increase without any overhead caused by switching to cupy.   
    That sounds really good.  
    Started to run the cifar10 notebook on the remote ligthning computer but it seems to run at the same speed with 14 epochs in 7 mins same as my computer...  
    At 14 mins, my computer actually outran the lightning AI computer: 32 epopchs vs 28 respectively.  
  - Turns out that cupy also has a warmup, let me test it a second time.
    Updated the code:
  ```python
  import numpy as np
  import cupy as cp
  import time

  # Define the input shapes
  views_shape = (10000, 26, 26, 3, 7, 7)
  k_shape = (32, 7, 7, 3)

  # Generate random input data
  views_np = np.random.rand(*views_shape)
  k_np = np.random.rand(*k_shape)

  # Convert to CuPy arrays
  views_cp = cp.array(views_np)
  k_cp = cp.array(k_np)

  # Warm-up CuPy
  _ = cp.tensordot(views_cp, k_cp, axes=([3, 4, 5], [3, 1, 2]))
  cp.cuda.Stream.null.synchronize()

  # Measure NumPy time
  start = time.time()
  tensordot_np = np.tensordot(views_np, k_np, axes=([3, 4, 5], [3, 1, 2]))
  numpy_time = time.time() - start
  print(f"NumPy tensordot Time: {numpy_time:.4f}s")

  # Measure CuPy time with multiple iterations
  iterations = 10
  start = time.time()
  for _ in range(iterations):
      _ = cp.tensordot(views_cp, k_cp, axes=([3, 4, 5], [3, 1, 2]))
  cp.cuda.Stream.null.synchronize()
  cp_time = (time.time() - start) / iterations
  print(f"Average CuPy tensordot Time: {cp_time:.4f}s")
  ```
  Ok, it's better:
  - numpy: 1.3s
  - cupy: 0.5s   
  I was expecting a bigger improvement tbh, like a 10x...  
  Let's see if uncleGPT can make it better.


03/04/2025:
  - Switching the batch size from 200 to 500 (tried to see if it would speed things up), made the model untrainable.  
    This modification was not pushed to the remote repo, maybe it did speed things up and that's why the remote lightning AI computer took more time to perform the epochs?  
    I'm running the training again with 200 batch size on my desktop let's see how that will change the speed.  
    No, it's, suprisingly enough, faster to train with smaller batch size: 33 epochs in 13 mins this time on my desktop.  
    Ok, I think I know why there was a speed diff, I also forgot the change to the architecture of the network.  
    Let's rerun the notebook on the remote computer to see if there is any diff.  
    At ~4 mins, it looks like it didin't change anything...  
    No diffs indeed...
  - Maybe I don't need to speed things up after all?
    Let's say it takes me ~2 days to get the best off of cupy and get a ~10x improvement it would be nice.  
    BUT, it might prevent someone who doesn't have a cuda GPU to run the notebook.  
    AND maybe I can just run the training on the 50ks which should take ~10x more time (5 hours) to train that give me ~two trained models per night.  
    That way I can focus on saving on the model.  
    Then I would move on to pytorch, I feel like I would be wasting time if I kept on building on top of numpy.
  - Let's try to scale up to 10k samples cifar10 subset, same architecture/hyperparameters.  
    First run did not succeed, I changed the batch size to 500 (to try to speed things up) and decreased the starting leargnig rate from 0.025 to 0.015.  
    It works!  
    80% accuracy at epoch 28 in ~22mins!  
    Weirdly enough, doubling the dataset size sped up the training...  
    Test accuracy raised to 40% test accuracy. 
  - Let's try the full dataset then (I might need to increase the swap)...
    I encountered this error: `MemoryError: Unable to allocate 37.0 GiB for an array with shape (50000, 26, 26, 3, 7, 7) and data type float64`.  
    Given the shape of the ndarray, I assum this is the ndarray of the views.  
    What's weird is that the sliding_window_view should only create a view of the input tensor.  
    Ok, looking at the traceback, I see that this is because the tensordot function uses a transpose on the views before performing the dot call.  
    This effectively requires copy (and therefore an allocation) of the views to be made.  
    Since this problem arises only for metric recording of the full dataset loss/accuracy I will simply split the forward call on the entire dataset in calls of 10k subsets of the dataset and then concat them.  
    That seems to have fixed it.
  - Three epochs and 4 mins in and we are already at 50% accuracy (on training set) I'm actually mind blown.  
    I wander if einsum would be faster... if it doesn't use transpose, it would not need to do all those allocations... to be continued.  
  - While the training continues I will get a beer...
  - OOOOOOOOOOOOOOOOOOO it fitted the full training dataset at 89% accuracy!!!!!!
  - Test accuracy is 50% but I think this is only because I left the training going on for too many epochs.
    I'll find out tomorrow

04/04/2025:
  - Interrupting the training at epoch 28, we get a train/test accuracy of 70 and 60 percent respectively.    
    This confirms that the model of yesterday did overfit.  
  - Trying to switch back to einsum to see if it speeds things up (looks like it doesn't).
    I tested them with `timeit` and tensordot is, in fact, faster.
  - Looked at a lot of videos on GPUs to see which one could fit me best, the 5070 or some GPU of the 30 series seem like the best options.  
  - I have also looked at cloud GPU options and runPod and Vast.ai seem pretty interesting.
  - Looked at the kernels and they resemble the cpu kaggle's kernels which is a good sign.

06/04/2025:
  - Retrained a model for 30 epochs with 15 kernels and interestingly enough, only 5 kernels are used.  
    By "not being used", I mean that the sum of the activation maps of a 1000 samples batch over the 0(sample), 1(width) and 2(height) axes are 0 after Relu.  
    I believe this is known as a dead neuron, although here it's more like a dead kernel.  
  - Note that I never(~5 trainings) experience this with 10 kernels models and always (with ~3 trainings) do with 15 kernels models.  
    So maybe this is due to a missimplementatin (in the batchNorm maybe)?
  - So let's try leakyRelu:manshrugging:.

07/04/2025:
  - LeakyRelu had a typo that made the gradient innacurate.
    After fixing it and switching Relu by LeakyRelu after the conv/BatchNorm layers, I trained the 15 kernels models for 22 epochs.  
    Got a training and test accuracy of 69 and 61 percent respectively.  
    This time all the filters are used.  
    I looked at the hidden Fc layer activations for 1000 imgs and the activations were also VERY sparse.  
    I will try replacing the second ReLu with LeakyReLu too.  

08/04/2025:
  - I ran a 35 epochs training on the LeakyRelu model but there was no improvement...

09/04/2025:
  - Tried l2 norm but it didn't seem to work...

10/04/2025:
  - Trying to use the LeakyRelu only on the hidden fc layer.
  - Saw in the kaggle notebook that the conv layer has a stride of 2 and a padding of 1, that something that might be worth looking into.  


15/04/2025:
  - Implemented a `sliding_window_view` on top of as_strided that works like the numpy function with the addition of the stride parameter.  
  - Implementing that function is just the tip of the iceberg tho.
    To update the entire network, I would also need to learn how the backward pass would change with strides and padding in the forward pass.  
    That's something I should look into but it would take some time I would rather look into regularization techniques like dropouts and take another look at l2 norm/weight decay.  
  - Watched [this video](https://www.youtube.com/watch?v=q7seckj1hwM&ab_channel=ArtemKirsanov) I didn't understand it entirely but it was imtersting and helped a little bit to understand l1 and l2 norm.

16/04/2025:
  - Made chatGPT implement a droput layer (booooh chatGPT I know, I know...)
  - Added dropout layer after maxpooling layer resulting in the following architecture:
    ```python
    [
        Convolutional((10, 7, 7, 3)),
        BatchNorm(),
        Relu(),
        MaxPool((2, 2)),
        Dropout(0.25),
        Flatten(),
        Linear(1690, 300),
        LeakyRelu(),
        Linear(300, y.shape[1]),
        Softmax(),
    ]
    ```
  - Started a training run for the night.

17/04/2025:
  - The test accuracy went to 61% so.... not that big of an improvement(sad).
  

24/04/2025:
  - I think this is going to be the end of this project, it's a shame I didn't get a better test accuracy.  
    I'm stopping here because I thnik I am spending time on waiting for the result of my improvement attemps.  
    I think I should have used something like `cupy` and implement an auto gradient computing.   
    Maybe I would get back to it later but now I need to start using `pytorch` and to learn high level concepts rather than wait for the training to finish...  
    And learn an actually used library.