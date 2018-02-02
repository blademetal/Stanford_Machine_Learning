function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
sums = 0;
for i=1:m
  %sums += power(((theta(1)*X(i,1)+theta(2)*X(i,2))-y(i)),2);
  sub_sum = 0;
  for j=1:size(X,2)
    sub_sum+=theta(j)*X(i,j);
  end
  sums += power((sub_sum-y(i)),2);
end

J = (1/(2*m))*sums;




% =========================================================================

end
