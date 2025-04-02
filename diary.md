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
  I'm going to look into CuPy, it also seems interesting