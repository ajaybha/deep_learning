# Variant Layers Deep Learning

This repository is based on two coursera on-line courses, [Neural Networks for Machine Learning](https://www.coursera.org/learn/neural-networks) by [Prof. Geoff Hinton](http://www.cs.toronto.edu/~hinton/) and [Machine Learning](https://www.coursera.org/learn/machine-learning/) by [Prof. Andrew Ng](http://www.andrewng.org/)
I tried to implement a function that is able to train variant hidden layers.

### How to use?

Training and prediction:

```
% training model
[Theta train_accuracy test_accuracy] = deep_learn( ...
    X, Xtest, ...   % training set and test set
    max_iter, ...   % maximum iteration for backpropagation
    max_iter_inner, ... % maximum iteration for CG
    max_iter_pre, ...   % maximum iteration for rbm
    batch_size, ...     % batch size for backpropagation
    batch_size_rbm, ... % batch size for rbm
    alpha, ...          % learning rate
    penalty, ...        % penalty
    penalty_rbm, ...    % L2 penalty for rbm pre-train
    m_type, ...     % momentum type (0: fixed momentum)(1: 0.5 before 5th iterations)(> 1: adaptive momentum with c1 = c2 = m_type)
    momentum, ...   % momentum
    reset_bias, ... % reset rbm bias or not: true|false
    pretrain, ...   % pretrain or not: true|false
    varargin);      % variant numbers of hidden layer, each one is number of hidden nodes

% predict other data set
pred = nnmPredict(data, Theta{:});
```

X, Xtest are cell arrays with number of labels elements, each element is a m*n matrix that has the same label, where m is number of data, n is number of feature. The output Theta is also a cell arry whish number of hidden layers plus one elements, each element is parameter matrix between two layers.

### Examples

In the examples folder, there is a MNIST example. To run the example, you've to download data set from [THE MNIST DATABASE](http://yann.lecun.com/exdb/mnist/) in advance.
