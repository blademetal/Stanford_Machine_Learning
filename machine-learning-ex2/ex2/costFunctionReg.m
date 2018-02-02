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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
sums = 0;
for i = 1:m
  sums += -y(i)*log(sigmoid(theta'*X(i,:)'))-(1-y(i))*log(1-sigmoid(theta'*X(i,:)'));
end

sum_lambda = 0;
for i = 2:size(X,2)
  sum_lambda += power(theta(i),2);
end

J = (1/m)*sums(1)+((lambda/(2*m))*sum_lambda);

theta_sums = [];
% if j=0
sub_sum = 0;
for k = 1:m
  sub_sum += (sigmoid(theta'*X(k,:)')-y(k))*X(k,1);
end
theta_sums(1) = (1/m)*sub_sum;


for j = 2:size(X,2)
  sub_sum = 0;
  for k = 1:m
    sub_sum += (sigmoid(theta'*X(k,:)')-y(k))*X(k,j);
  end
  theta_sums(j) = ((1/m)*sub_sum)+((lambda/(m))*theta(j));
end

for i=1:size(X,2)
  grad(i) = theta_sums(i);
end





% =============================================================

end
