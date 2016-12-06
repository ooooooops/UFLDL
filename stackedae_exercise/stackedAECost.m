function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%
numCases = M;
%1.forward propagate to get P
active_L1 = sigmoid(stack{1}.w*data+repmat(stack{1}.b,1,size(data,2)));
active_L2 = sigmoid(stack{2}.w*active_L1+repmat(stack{2}.b,1,size(data,2)));
M = softmaxTheta*active_L2;
M = bsxfun(@minus, M, max(M, [], 1));
M = exp(M);
%fz = M;
P = bsxfun(@rdivide, M, sum(M));
%2.backward propagate to get residual
%out_residual = -(groundTruth-P)*active_L2'.*fz;
L2_residual = -softmaxTheta'*(groundTruth-P).*(active_L2.*(1-active_L2));
%L2_residual = softmaxTheta'*out_residual.*(active_L2.*(1-active_L2));
L1_residual = stack{2}.w'*L2_residual.*(active_L1.*(1-active_L1));

softmaxThetaGrad = (groundTruth-P)*active_L2'/(-numCases) + lambda*softmaxTheta;

stackgrad{2}.w = stackgrad{2}.w + L2_residual*active_L1';
stackgrad{2}.b = stackgrad{2}.b + sum(L2_residual,2);
stackgrad{1}.w = stackgrad{1}.w + L1_residual*data';
stackgrad{1}.b = stackgrad{1}.b + sum(L1_residual,2);

stackgrad{2}.w = stackgrad{2}.w/numCases;% + lambda*stack{2}.w;
stackgrad{2}.b = stackgrad{2}.b/numCases;
stackgrad{1}.w = stackgrad{1}.w/numCases;% + lambda*stack{1}.w;
stackgrad{1}.b = stackgrad{1}.b/numCases;

%3.cost
cost = groundTruth(:)'*log(P(:))/(-numCases) + sum((softmaxTheta(:)).^2)*lambda/2;
%cost = groundTruth(:)'*log(P(:))/(-numCases) + sum((softmaxTheta(:)).^2)*lambda/2;
% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
