function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
n = length(theta);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h = sigmoid(X * theta); % mx1 vector


for i = 1:m
	J = J + y(i)* log(h(i)) + (1-y(i))*log(1-h(i));
end
J = J / m * -1;
% from 2 to n
for j = 2: n
	J = J + lambda / (2 * m) * theta(j) * theta(j) ;
end
% caculate the gradient
for i = 1: m
	grad = grad + (h(i)- y(i))* X(i, :)' ;
end
grad = grad / m ;

gradWithRegulariztion = [0;theta(2:end)] * lambda / m;
grad = grad + gradWithRegulariztion ;

% =============================================================

end
