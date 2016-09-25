function [Theta Hiden] = rbmBD(X, hidden_layer_size, penalty, alpha, m_type, momentum, reset_bias, batch_size, max_iter)
%rbm Implements RBM
%
%       X: visible nodes
%       hidden_layer_size: number of hidden layer nodes
%       penalty: L2 penality
%       alpha: learning rate
%       m_type: 0: fix moment, 1: 0.5 in the beginning, 2: adaptive
%       momentum: final momentum
%       reset_bias: true | false, reset bias or not
%       batch_size: number of data set in a batch
%       max_iter: maximum iteration
%   return value:
%       Theta: Visible X Hidden
%

    num_labels = length(X);
    num_sample = 0;
    for i = 1 : num_labels
        num_sample = num_sample + size(X{i}, 1);
    end
    num_batch = ceil(num_sample/batch_size);

    num_feature = size(X{1}, 2);

    cat_batch_size = cell(num_labels, 1);
    for i = 1 : num_labels
        cat_batch_size{i} = ceil(size(X{i}, 1)/num_batch);
    end

    momentum_org = momentum;
    if m_type == 1
        momentum = 0.5; % initial momentum for first 5 iteration
    end

    % initialize Theta with 0 mean, 0.1 standard deviation in Gaussian distribution
    Theta = 0.1 * randn(num_feature, hidden_layer_size);

    % add bias for visible layer
    Theta = [zeros(1, size(Theta, 2)); Theta];
    % add bias for hidden layer
    Theta = [zeros(size(Theta, 1), 1) Theta];

    % for momentum
    Velocity = zeros(num_feature + 1, hidden_layer_size + 1);

    anneal = true;
    penalty_org = penalty;
    c1 = m_type;
    c2 = c1;

    for i = 1 : max_iter

        %sqr_err = 0;
        if m_type == 1
            if i > 5
                momentum = momentum_org;
            end
        elseif m_type > 1
            momentum = momentum_org * c1 / ((max_iter - i) + c2);
        end

        if penalty > 0
            penalty = penalty_org * c1 / ((i - 1) + c2);
        end

        for batch = 1 : num_batch
            % partition mini-batch
            Xbatch = [];
            for cat = 1 : num_labels
                start_row = cat_batch_size{cat} * (batch - 1) + 1;
                end_row = cat_batch_size{cat} * batch;
                if end_row > size(X{cat}, 1)
                    end_row = size(X{cat}, 1);
                end
                Xbatch = [Xbatch; X{cat}(start_row:end_row, :)];
            end
            num_rows = size(Xbatch, 1);
            if num_rows == 0
                break;
            end
            Xbatch = shuffle(Xbatch);

            % Positive phase
            Vpos = [ones(num_rows, 1) Xbatch];
            Vpos_state = Vpos > (0.1 * randn());
            Hpos = sigmoid(Vpos * Theta);
            Hpos_state = Hpos > rand(num_rows, hidden_layer_size + 1);

            % Negative phase (reconstruction)
            Vneg = sigmoid(Hpos_state * Theta');
            if reset_bias
                % bias of Visible should be always 1.
                Vneg(:, 1) = ones(num_rows, 1);
            end
            Hneg = sigmoid(Vneg * Theta);

            % update Theta (TODO: learning from noise visible nodes)
            delta = (Vpos'*Hpos - Vneg'*Hneg)/num_rows;

            if penalty > 0
                % penalty (should not apply to bias)
                biasV = Theta(1, :);
                biasH = Theta(:, 1);
                Theta -= (penalty * alpha/num_rows * Theta);
                Theta(1, :) = biasV;
                Theta(:, 1) = biasH;
            end

            % add momentum and update theta
            Velocity = (Velocity * momentum + alpha * delta);
            Theta += Velocity;
            %sqr_err += (sum(sum((Vpos - Vneg).^2)));
        end
        %fprintf('RBM total square error of iteration %4i/%4i: %4.6e\r', i, max_iter, sqr_err);

    end

    Hiden = cell(num_labels, 1);
    for cat = 1 : num_labels
        Hiden{cat} = sigmoid([ones(size(X{cat}, 1), 1) X{cat}] * Theta);
        % remove bias
        Hiden{cat} = Hiden{cat}(:, 2 : end);
    end
end
