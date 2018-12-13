function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

h=sigmoid(X*theta);
J=sum((-y.*log(h))-((1-y).*log(1-h)))/m+sum(theta(2:end).^2)*(lambda/(2*m));
%J = sum(-y .* log(H) - (1 - y) .* log(1 - H)) / m;
b=h-y;
grad=(X'*b)/m;
temp=theta;
temp(1)=0;
grad = grad(:)+(lambda/m)*temp;

end
