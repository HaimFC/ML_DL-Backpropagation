# ML_DL Assignment 2 - Explanation and Solution

## Overview of the Assignment
This assignment focuses on implementing and understanding a logistic regression model for binary classification. The key aspects involve:
1. Calculating predictions using the sigmoid function.
2. Implementing a cost function based on cross-entropy loss.
3. Computing gradients for optimization.
4. Evaluating the model's accuracy.

### Objectives:
- Understand the mathematical foundations of logistic regression.
- Implement logistic regression components programmatically.
- Train the model using gradient descent to find optimal weights and bias.

---

## Solution Description

### 1. Logistic Regression
Logistic regression is used for binary classification, where the target variable \( t \) can take values 0 or 1. The model predicts a probability \( y \) using the sigmoid function:
\[
y = \sigma(w \cdot X^T + b)
\]
where:
- \( w \) is the weight vector,
- \( b \) is the bias,
- \( X \) is the input matrix.

---

### 2. Functions Implemented

#### **`sigmoid(z)`**
**Description:**  
Applies the sigmoid function to the input \( z \) to compute probabilities:
\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

**Usage:**  
This is the activation function used for logistic regression.

---

#### **`cross_entropy(t, y)`**
**Description:**  
Calculates the cross-entropy loss for binary classification:
\[
L(t, y) = -t \log(y) - (1 - t) \log(1 - y)
\]

**Usage:**  
Used as the loss function for training logistic regression models.

---

#### **`pred(w, b, X)`**
**Description:**  
Generates predictions for the input data \( X \) using the weights \( w \) and bias \( b \).

**Preconditions:**  
- \( w \): Weight vector, shape (90,).  
- \( b \): Bias term, scalar.  
- \( X \): Input matrix, shape (N, 90).

**Returns:**  
- \( y \): Predicted probabilities, shape (N,).

---

#### **`cost(y, t)`**
**Description:**  
Computes the average cross-entropy loss between predictions \( y \) and true labels \( t \):
\[
L = \frac{1}{N} \sum_{i=1}^{N} L(t_i, y_i)
\]

**Usage:**  
Measures the performance of the model during training.

---

#### **`derivative_cost(X, y, t)`**
**Description:**  
Computes the gradients of the loss function with respect to weights \( w \) and bias \( b \):
\[
\frac{\partial L}{\partial w} = \frac{1}{N} X^T (y - t)
\]
\[
\frac{\partial L}{\partial b} = \frac{1}{N} \sum_{i=1}^{N} (y_i - t_i)
\]

**Returns:**  
- \( dLdw \): Gradient with respect to weights, shape (90,).  
- \( dLdb \): Gradient with respect to bias, scalar.

---

#### **`get_accuracy(y, t)`**
**Description:**  
Calculates the classification accuracy:
\[
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Predictions}}
\]

**Usage:**  
Evaluates the model's performance on validation data.

---

## Optimal Weights and Bias
The file `assignment2_submission_optimal_weights.npy` contains the optimized weight vector \( w \), and `assignment2_submission_optimal_bias.npy` contains the optimized bias \( b \). These values were calculated by training the logistic regression model using gradient descent, minimizing the cross-entropy loss.

---

## Summary
This assignment provided a hands-on approach to implementing logistic regression. By breaking the process into functions, the problem was solved step-by-step:
1. Sigmoid activation was applied for predictions.
2. Cross-entropy loss was computed to measure performance.
3. Gradients were calculated to update weights and bias during training.
4. Accuracy was used to evaluate the model's success.

The optimal weights and bias were submitted after training the model.
