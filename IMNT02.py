import numpy as np
import pandas as pd
from math import exp

# Load data from CSV file into a DataFrame
df = pd.read_csv("IMNT_DATASET.csv")
# Extracting the 'Tasks', 'Bugs', and 'Issues' columns as feature attributes
attributes = df[['Tasks', 'Bugs', 'Issues']].values
# Extracting the 'Class' column as class labels
classes = df['Class'].values
# Getting unique class values
unique_classes = np.unique(classes)

# Normalize the attributes
# Subtracting mean (axis=0 ensures the mean of each column is taken) 
# and dividing by standard deviation for each column respectively
normalized_attributes = (attributes - attributes.mean(axis=0)) / attributes.std(axis=0)

# Shuffling data and splitting it into training and testing datasets
indices = np.arange(len(attributes))
np.random.shuffle(indices)

train_size = int(0.8 * len(attributes))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

train_attributes = normalized_attributes[train_indices]
train_classes = classes[train_indices]

test_attributes = normalized_attributes[test_indices]
test_classes = classes[test_indices]

# Defining the perceptron model
lr = 0.2

class Perceptron:
    def __init__(self, input_size: int):
        self.w = np.zeros(input_size)  # Initialize weights
        self.w0 = 0  # Bias term
        self.grad = 0  # Gradient
    
    def predict(self, input_values):
        # Calculate the weighted sum
        net = np.dot(input_values, self.w) + self.w0
        # Calculate the gradient using derivative of sigmoid
        self.grad = self.der_sigmoid(net)
        # Return the predicted value (between 0 and 1)
        return self.act_sigmoid(net)

    def fit(self, input_values, output_value):
        # Predict the output for given input values
        y = self.predict(input_values)
        # Calculate the error
        delta = output_value - y
        if delta != 0:
            # Adjust weights and bias based on the error
            for i in range(len(self.w)):
                self.w[i] += lr * delta * self.grad * input_values[i]
                self.w0 += lr * delta * self.grad

    # Activation function: Sigmoid
    def act_sigmoid(self, val):
        return 1.0 / (1.0 + exp(-val))

    # Derivative of the sigmoid function
    def der_sigmoid(self, val):
        s = self.act_sigmoid(val)
        return s * (1 - s)
    

# Training perceptrons for each unique class
perceptrons = [Perceptron(3) for _ in unique_classes]
for epoch in range(1000):
    for di in range(len(train_attributes)):
        target_class = train_classes[di]
        for i, uc in enumerate(unique_classes):
            expected_output = 1 if target_class == uc else 0
            perceptrons[i].fit(train_attributes[di], expected_output)

# Testing perceptrons and printing results
for di in range(len(test_attributes)):
    outputs = [perc.predict(test_attributes[di]) for perc in perceptrons]
    predicted_class = unique_classes[np.argmax(outputs)]
    original_data = attributes[test_indices[di]]
    print(f"Object: {original_data}, Real class: {test_classes[di]}, Predicted class: {predicted_class}")

# Function to calculate RMSE and number of misclassifications
def calculate_errors(attributes, classes, perceptrons):
    sum_squared_errors = 0
    misclassification_count = 0
    for di in range(len(attributes)):
        outputs = [perc.predict(attributes[di]) for perc in perceptrons]
        predicted_class = unique_classes[np.argmax(outputs)]
        actual_class = classes[di]
        # Sum squared errors for RMSE
        sum_squared_errors += (predicted_class - actual_class)**2
        # Count misclassifications
        if predicted_class != actual_class:
            misclassification_count += 1
    # Calculate RMSE
    rmse = np.sqrt(sum_squared_errors / len(attributes))
    return rmse, misclassification_count

# Calculating and printing RMSE and number of misclassifications for training and testing datasets
train_rmse, train_misclassified_count = calculate_errors(train_attributes, train_classes, perceptrons)
test_rmse, test_misclassified_count = calculate_errors(test_attributes, test_classes, perceptrons)

print("\nTraining Data Metrics:")
print(f"RMSE: {train_rmse:.3f}")
print(f"Misclassified: {train_misclassified_count} out of {len(train_attributes)}")

print("\nTesting Data Metrics:")
print(f"RMSE: {test_rmse:.3f}")
print(f"Misclassified: {test_misclassified_count} out of {len(test_attributes)}")

# Making a prediction for a new data point
new_data = np.array([2, 3, 2])
# Normalize the new data using the mean and standard deviation of the original dataset
normalized_new_data = (new_data - attributes.mean(axis=0)) / attributes.std(axis=0)
# Predict the class for the new data point
outputs = [perc.predict(normalized_new_data) for perc in perceptrons]
predicted_class = unique_classes[np.argmax(outputs)]

print(f"For input {new_data}, predicted class is {predicted_class}")