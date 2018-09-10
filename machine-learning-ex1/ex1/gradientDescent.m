function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

x1 = X(:,1);
x2 = X(:,2);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    delta1 = 0;
    delta2 = 0;

    for i = 1:m
        delta1 = delta1 + alpha*(theta(1)*x1(i) + theta(2)*x2(i) - y(i))/m;
        delta2 = delta2 + alpha*((theta(1)*x1(i) + theta(2)*x2(i) - y(i))*x2(i))/m;
    end

    theta(1) = theta(1) - delta1;
    theta(2) = theta(2) - delta2;

    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

end

end
