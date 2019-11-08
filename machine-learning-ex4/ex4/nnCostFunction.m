function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
y_new = zeros(size(y),num_labels) ;
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
a1 = [ones(size(X),1) X] ;
z2 = a1*Theta1' ;
a2 = sigmoid(z2) ; 
a2 = [ones(size(a2),1) a2] ;
z3 = a2*Theta2' ;
a3 = sigmoid(z3) ;
for i = 1:num_labels ;
	y_new(:,i) = (y == i) ;   % y_new(i,k)
endfor
% a3(i, k)
% without regularization
%J = (1./m)*trace(-log(a3)*y_new'-log(1.-a3)*(1.-y_new)') ; 
% with regularization
J = (1./m)*trace(-log(a3)*y_new'-log(1.-a3)*(1.-y_new)') ...
     + (lambda/(2*m))*(sum(nn_params.^2)-sum(Theta1(:,1).^2)-sum(Theta2(:,1).^2)) ; 


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
D_1 = 0 ;
D_2 = 0 ;
for t = 1:m ;
   a_1 = X(t,:) ;
   a_1 = [1, a_1] ; % adding ones [1 401]
   z_2 = a_1*Theta1' ;  % [1 401] * [401 25] = [1 25]
   a_2 = sigmoid(z_2) ; 
   a_2 = [1, a_2]; % [1 26] 
   z_3 = a_2*Theta2' ; %[1 10]
   a_3 = sigmoid(z_3) ;

   delta_3 = a_3 - y_new(t,:) ; % [1 k] dimension where k = 10 
   tmp_delta2 = delta_3*Theta2 ; % [1 10] * [10 26] = [1 26]
   delta_2 = tmp_delta2(2:end).*sigmoidGradient(z_2) ; % [1 25]
   D_1 = D_1 + a_1'*delta_2 ;  %[401 25]
   D_2 = D_2 + a_2'*delta_3 ;  %[26 10]  

%Theta1_grad = (1/m)*D_1' ;
%Theta2_grad = (1/m)*D_2' ;
Theta1_grad = (1/m)*D_1'+(lambda/m) * Theta1_grad ;
Theta2_grad = (1/m)*D_2'+(lambda/m) * Theta2_grad ;



%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:,1) = (1/m)*D_1'(:,1) ; %  Theta1_grad(:,1)- (lambda/m)*Theta1_grad(:,1) ;
Theta2_grad(:,1) = (1/m)*D_2'(:,1) ; %  Theta2_grad(:,1)- (lambda/m)*Theta2_grad(:,1) ;
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
