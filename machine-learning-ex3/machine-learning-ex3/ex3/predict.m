function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1); % X : 5000 * 400
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
X = [ones(m, 1) X]; % add 1 
% Theta1:25 x 401  X' : 401 x 5000  
% z2 : 25 * 5000
z2 = Theta1 * X'; 

% Theta2 : 10 * 26
a2 = sigmoid(z2);
a2 = a2';
% a2 : 5000*25

a2 = [ones(size(a2, 1),1) a2];

z3 = Theta2 * a2';
h = sigmoid(z3);
[Y, p] = max(h, [] , 1);
p = p' ;








% =========================================================================


end
