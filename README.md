# CS165-project
Caltech CS165 final：fully deterministic neural network with deterministic initialization and non-stochastic full-batch training    

In this work, we aim to provide a successful counterexample to prove that such stochasticity is NOT necessary for ML theory.
We apply a fully deterministic initialization scheme as well as non-stochastic full-batch training, which initializes the weights of networks with only zeros and ones and explicit regularization during model training. Our work proposes that random weights and SGD may be unnecessary for neural networks, and it is possible to train neural networks without any randomness while achieving state-of-the-art performance.  


This project is based on Jiawei Zhao and Jonas Geiping's previous publications and work:  
https://arxiv.org/abs/2110.12661  
https://arxiv.org/abs/2109.14119
