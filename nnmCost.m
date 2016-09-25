function [J grad] = nnmCost(nn_params, ...
                            X, y, penalty, ...
                            input_layer_size, ...
                            num_labels, ...
                            varargin)
%NNMCOST Implements the multiple layers neural network cost function
%which performs classification
%   [J grad] = nnmCost(nn_params, X, y, penalty, input_layer_size, num_labels, ...
%   hidden_layer1_size, ..., hidden_layerN_size) computes the cost and gradient
%   of the neural network. The parameters for the neural network are "unrolled"
%   into the vector nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad is an "unrolled" vector of the partial derivatives
%   of the neural network.
%

    num_hidden_layers = length(varargin);
    Theta = cell(num_hidden_layers + 1, 1);

    offset = 1;
    hidden_layer_size = varargin{1};
    param_size = hidden_layer_size * (input_layer_size + 1);
    Theta{1} = reshape(nn_params(offset : offset + param_size - 1), ...
                    hidden_layer_size, (input_layer_size + 1));


    hidden_layer_size_prev = hidden_layer_size;
    for i = 2 : num_hidden_layers
        offset = offset + param_size;
        hidden_layer_size = varargin{i};
        param_size = hidden_layer_size * (hidden_layer_size_prev + 1);
        Theta{i} = reshape(nn_params(offset : offset + param_size - 1), ...
                            hidden_layer_size, (hidden_layer_size_prev + 1));
        hidden_layer_size_prev = hidden_layer_size;
    end

    offset = offset + param_size;
    Theta{num_hidden_layers + 1} = reshape(nn_params(offset : end), ...
                                        num_labels, (hidden_layer_size_prev + 1));

    % Setup some useful variables
    m = size(X, 1);

    Theta_grads = cell(num_hidden_layers + 1, 1);
    X_plus1 = [ones(m,1) X];

    % Regularization term of cost
    Jreg = 0;

    for i = 1 : num_hidden_layers + 1
        Theta_grads{i} = zeros(size(Theta{i}));
        Jreg = Jreg + sum(sum(Theta{i}(:, 2:end).^2));
    end

    % Feedforward pass
    Z = cell(num_hidden_layers + 1, 1);
    A = cell(num_hidden_layers + 1, 1);
    Z{1} = Theta{1} * X_plus1';
    A{1} = sigmoid(Z{1});
    for i = 2 : num_hidden_layers + 1
        Z{i} = Theta{i} * [ones(1,m); A{i-1}];
        A{i} = sigmoid(Z{i});
    end

    % Labels
    Y = ((ones(m, 1) * [1:num_labels]) == y)';

    % Cost
    J = sum(sum(-Y.* log(A{num_hidden_layers + 1}) - (1-Y).*log(1 - A{num_hidden_layers + 1})))/m;

    % w/ regularization
    J = J + penalty * Jreg / (2*m);

    % Backpropagation pass

    % output layer
    delta = A{num_hidden_layers + 1} - Y;
    Theta_grads{num_hidden_layers + 1} = Theta_grads{num_hidden_layers + 1} + delta * [ones(1, m); A{num_hidden_layers}]';

    % hidden layer
    delta = Theta{num_hidden_layers + 1}' * delta.* sigmoidGradient([ones(1, m); Z{num_hidden_layers}]);
    for j = num_hidden_layers : -1 : 2
        Theta_grads{j} = Theta_grads{j} + delta(2:end,:) * [ones(1, m); A{j-1}]';
        delta = Theta{j}' * delta(2:end,:).* sigmoidGradient([ones(1, m); Z{j-1}]);
    end
    Theta_grads{1} = Theta_grads{1} + delta(2:end, :) * X_plus1;

    grad = [];
    for i = 1 : num_hidden_layers + 1
        Theta_grads{i} = Theta_grads{i}/m;

        % w/ regularization
        Theta_grads{i}(:, 2:end) = Theta_grads{i}(:, 2:end) + penalty*Theta{i}(:, 2:end)/m;

        grad = [grad; Theta_grads{i}(:)];
    end

end
