function p = nnmPredict(X, varargin)
%NNMPREDICT Predict the label of an input given a trained neural network
%   p = nnmPredict(X, Theta1, ..., ThetaN) outputs the predicted label of
%   X given the trained weights of a neural network (Theta1, ..., ThetaN)

    m = size(X, 1);
    p = zeros(size(X, 1), 1);
    
    h = X;
    for i = 1 : length(varargin)
        Theta = varargin{i};
        h = sigmoid([ones(m, 1) h] * Theta');
    end
    
    [dummy, p] = max(h, [], 2);

% =========================================================================

end
