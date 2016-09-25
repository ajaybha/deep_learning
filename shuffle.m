function sx = shuffle(x)
%SHUFFLE randomize matrix
%

[dummy idx] = sort(rand(size(x, 1), 1));
sx = x(idx, :);
end