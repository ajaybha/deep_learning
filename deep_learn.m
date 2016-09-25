function [Theta train_accuracy test_accuracy] = deep_learn( ...
    X, Xtest, ...
    max_iter, max_iter_inner, max_iter_rbm, ...
    batch_size, batch_size_rbm, ...
    alpha, penalty, penalty_rbm, ...
    m_type, momentum, ...
    reset_bias, pretrain, varargin)
%DEEP_LEARN Learning multilayer neural network model which performs classification
%   [Theta train_accuracy test_accuracy] = DEEP_LEARN(X, Xtest, max_iter, ...
%   max_iter_inner, max_iter_rbm, batch_size, batch_size_rbm, alpha, ...
%   penalty, penalty_rbm, m_type, momentum, reset_bias, pretrain, varargin)
%   learns multilayer neural network model and computes accuracy of training set
%   and test set.
%
%   arguments:
%       X, Xtest: train data set and test data set, both are arrays with the
%           numbers of label elements, in other words, each label of data set
%           (a matrix) is an array element.
%       max_iter, max_iter_inner, max_iter_rbm: maximum iteration for
%           backpropagation, batch CG, and rbm pre-train respectively.
%       batch_size, batch_size_rbm: batch size for backpropagation and
%           rbm pre-train respectively.
%       alpha: learning rate.
%       penalty: L2 penalty.
%       penalty_rbm: L2 penalty for rbm pre-train.
%       m_type: momentum type
%           0: fixed momentum
%           1: 0.5 before 5 iterations
%           > 1: adaptive momentum with c1 = c2 = m_type
%       momentum: rbm gradient momentum.
%       pretrain: true | false, do pretrain or not.
%       reset_bias: true | false, reset bias for every rbm iteration
%       varargin: one or more hidden layer size.
% 
%   return value:
%       Theta: multilayer neural network model parameters, is an array, each
%           layer's parameters (a matrix) is an element.
%       train_accuracy: accuracy of training data set.
%       test_accuracy: accuracy of test data set.
%

    num_labels = length(X);
    num_sample = 0;
    for i = 1 : num_labels
        num_sample = num_sample + size(X{i}, 1);
    end
    num_batch = ceil(num_sample/batch_size);

    cat_batch_size = zeros(num_labels, 2); % col_1 = fix(), col_2 = mod()
    for i = 1 : num_labels
        cat_batch_size(i, 1) = fix(size(X{i}, 1)/num_batch);
        cat_batch_size(i, 2) = mod(size(X{i}, 1), num_batch);
    end

    input_layer_size  = size(X{1}, 2);
    num_hidden_layers = length(varargin);
    hidden_layer_size = cell(num_hidden_layers, 1);

    for i = 1 : num_hidden_layers
        hidden_layer_size{i} = varargin{i};
    end

    init_nn_params = [];
    if pretrain
        data_in = X;
        for i = 1 : num_hidden_layers
            [Theta_pretrain data_in] = rbmBD(...
                data_in, ...
                hidden_layer_size{i}, ...
                penalty_rbm, alpha, ...
                m_type, momentum, ...
                reset_bias, batch_size_rbm, max_iter_rbm);
            init_nn_params = [init_nn_params; Theta_pretrain(:, 2:end)'(:)];
        end
    else
        x_len = input_layer_size;
        for i = 1 : num_hidden_layers
            Theta_init = randInitializeWeights(x_len, hidden_layer_size{i});
            x_len = hidden_layer_size{i};
            init_nn_params = [init_nn_params; Theta_init(:)];
        end
    end
    init_nn_params = [init_nn_params; randInitializeWeights(hidden_layer_size{num_hidden_layers}, num_labels)(:)];

    options = optimset('MaxIter', max_iter_inner);

    c1 = 1;
    c2 = c1;
    penalty_org = penalty;

    for epoch = 1 : max_iter

        fprintf('\nBackpropagation iteration: %4i/%4i\n', epoch, max_iter);
        penalty = penalty_org * c1 / ((epoch - 1) + c2);

        for batch = 1 : num_batch
            % partition mini-batch
            Xbatch = [];
            for cat = 1 : num_labels
                tmp_batch_size = cat_batch_size(cat, 1);
                if batch <= cat_batch_size(cat, 2)
                    tmp_batch_size = tmp_batch_size + 1;
                end
                start_row = tmp_batch_size * (batch - 1) + 1;
                if batch > cat_batch_size(cat, 2)
                    start_row = start_row + cat_batch_size(cat, 2);
                end
                end_row = start_row + tmp_batch_size - 1;
                Xbatch = [Xbatch; [cat*ones(end_row - start_row + 1, 1) X{cat}(start_row:end_row, :)]];
            end
            num_rows = size(Xbatch, 1);
            Xbatch = shuffle(Xbatch);

            costFunction = @(p) nnmCost(p, ...
                                        Xbatch(:, 2:end), Xbatch(:, 1), ...
                                        penalty, input_layer_size, num_labels, ...
                                        varargin{:});

            [nn_params, cost] = fmincg(costFunction, init_nn_params, options);

             % Obtain Theta back from nn_params, and prepare init_nn_params for next iter
            init_nn_params = [];
            offset = 1;
            prev_layer_size = input_layer_size;
            for i = 1 : num_hidden_layers
                param_size = hidden_layer_size{i} * (prev_layer_size + 1);
                Theta{i} = reshape(nn_params(offset : offset + param_size - 1), ...
                                hidden_layer_size{i}, (prev_layer_size + 1));
                init_nn_params = [init_nn_params; Theta{i}(:)];

                % next offset
                offset = offset + param_size;
                prev_layer_size = hidden_layer_size{i};
            end
            Theta{num_hidden_layers + 1} = reshape(nn_params(offset : end), ...
                                                num_labels, (prev_layer_size + 1));
            init_nn_params = [init_nn_params; Theta{num_hidden_layers + 1}(:)];
        end
    end

    num_test = 0;
    test_labels = size(Xtest, 1);
    correct_train = 0;
    correct_test = 0;
    for cat = 1 : num_labels
        pred = nnmPredict(X{cat}, Theta{:});
        correct_train = correct_train + sum(pred == cat);
        if test_labels == num_labels && !isempty(Xtest{cat})
            pred_test = nnmPredict(Xtest{cat}, Theta{:});
            correct_test = correct_test + sum(pred_test == cat);
            num_test = num_test + size(Xtest{cat}, 1);
        end
    end

    train_accuracy = double(correct_train) * 100 / num_sample;
    if num_test > 0
        test_accuracy = double(correct_test) * 100 / num_test;
    end

% =========================================================================

end

