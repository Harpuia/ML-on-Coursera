function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
C_vec = [1 3 10 30 100 300 1000 3000];
sigma_vec = [0.3 1 3 10 30 100 300 1000];
error_val = zeros(length(C_vec), length(sigma_vec));
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

for k = 1:length(C_vec)
    for l = 1:length(sigma_vec)
        model = svmTrain(X, y, C_vec(k), @(x1, x2) gaussianKernel(x1, x2, sigma_vec(l)));
        predictions = svmPredict(model, Xval);
        error_val(k, l) = mean(double(predictions ~= yval));
    end
end
[error, index_row] = min(error_val);
[min_error, index_column] = min(error);
C = C_vec(index_row);
sigma = sigma_vec(index_column);





% =========================================================================

end
