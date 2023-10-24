from matplotlib import pyplot as plt
import numpy as np
from math import exp
import pandas as pd

w0 = 4.5
w1 = 0.5
w2 = 1

def perc_out(x1, x2):
    z = -1 * w0 + x1 * w1 + x2 * w2
    y = 1 / (1 + exp(-1 * z))
    return y

def act_step(val):
    if val > 0:
        return 1
    else:
        return 0

def der_step(val):
    return 1

def act_sigmoid(val):
    return 1.0 / (1.0 + exp(-val))

def der_sigmoid(val):
    s = act_sigmoid(val)
    return s * (1 - s)

lr = 0.05

class Perceptrons:
    def __init__(self, input_size: int):
        self.w = np.zeros(input_size)
        self.w0 = 0
        self.grad = 0
    
    def predict(self, input_values):
        net = np.dot(input_values, self.w)
        net += self.w0
        self.grad = der_sigmoid(net)
        return act_sigmoid(net)

    def fit(self, input_values, output_value):
        y = self.predict(input_values)
        delta = output_value - y
        if delta != 0:
            for i in range(len(self.w)):
                self.w[i] += lr * delta * self.grad * input_values[i]
                self.w0 += lr * delta * self.grad

# Class 0 data
train = pd.read_csv("IMNT_DATASET_CLASS0.csv")
train_classes = pd.read_csv("IMNT_DATASET_CLASS0.csv")
train = train.drop(columns=['Class', 'Issues'])
train_classes = train_classes.drop(columns=['Tasks','Bugs','Issues'])

# Class 1 data
train_1 = pd.read_csv("IMNT_DATASET_CLASS1.csv")
train_classes_1 = pd.read_csv("IMNT_DATASET_CLASS1.csv")
train_1 = train_1.drop(columns=['Class', 'Issues'])
train_classes_1 = train_classes_1.drop(columns=['Tasks','Bugs','Issues'])

# Divide data into test and train data with a ratio of 20 : 80
test_rows_to_select = int(0.2 * len(train))
train_rows_to_select = int(0.8 * len(train))

# Filling test data arrays.
d1_x1 = train.drop(columns=['Bugs']).to_numpy()[:test_rows_to_select]
d1_x2 = train.drop(columns=['Tasks']).to_numpy()[:test_rows_to_select]

d2_x1 = train_1.drop(columns=['Bugs']).to_numpy()[:test_rows_to_select]
d2_x2 = train_1.drop(columns=['Tasks']).to_numpy()[:test_rows_to_select]

dati = np.vstack((train.to_numpy()[:train_rows_to_select], train_1.to_numpy()[:train_rows_to_select]))

classes = np.vstack((train_classes[:train_rows_to_select], train_classes_1[:train_rows_to_select])).flatten().tolist()

perc = Perceptrons(2)

for epoch in range(1000):
    for di in range(len(dati)):
        perc.fit(dati[di], classes[di])

img = np.zeros((121, 121))

for yi in range(121):
    for xi in range(121):
        img[yi][xi] = perc.predict([xi / 10, yi / 10])

plt.scatter(d1_x1, d1_x2, s=70, c='red', marker='o', linewidths=1, edgecolors='black', label='Class 0 (Easy Sprint)')
plt.scatter(d2_x1, d2_x2, s=70, c='green', marker='o', linewidths=1, edgecolors='black', label='Class 1 (Medium Sprint)')
plt.imshow(img, extent=[0, 12, 0, 12], origin='lower')

plt.title('Class 0 vs Class 1 clusterization')
plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))

## THIS IS DONE BY ChatGPT
# Initialize variables to store error metrics 
train_rmse = 0
test_rmse = 0
train_misclassified = 0
test_misclassified = 0

# Calculate error metrics for training data
for di in range(len(dati)):
    predicted_class = perc.predict(dati[di])
    if classes[di] == 0:
        train_rmse += (predicted_class - 0) ** 2
        if round(predicted_class) != 0:
            train_misclassified += 1
    elif classes[di] == 1:
        train_rmse += (predicted_class - 1) ** 2
        if round(predicted_class) != 1:
            train_misclassified += 1

train_rmse = np.sqrt(train_rmse / len(dati))

# Calculate error metrics for test data (by GPT)
for di in range(len(d1_x1)):
    predicted_class = perc.predict([d1_x1[di][0], d1_x2[di][0]])
    test_rmse += (predicted_class - 0) ** 2
    if round(predicted_class) != 0:
        test_misclassified += 1

for di in range(len(d2_x1)):
    predicted_class = perc.predict([d2_x1[di][0], d2_x2[di][0]])
    test_rmse += (predicted_class - 1) ** 2
    if round(predicted_class) != 1:
        test_misclassified += 1

test_rmse = np.sqrt(test_rmse / (len(d1_x1) + len(d2_x1)))

print("CLASS 1 vs CLASS 0")
print("Training RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
print("Training Misclassified Count:", train_misclassified)
print("Test Misclassified Count:", test_misclassified)

plt.show()

#################################################################################################################################

# Class 0 data
train = pd.read_csv("IMNT_DATASET_CLASS1.csv")
train_classes = pd.read_csv("IMNT_DATASET_CLASS1.csv")
train = train.drop(columns=['Class', 'Issues'])
train_classes = train_classes.drop(columns=['Tasks','Bugs','Issues'])

# Class 1 data
train_1 = pd.read_csv("IMNT_DATASET_CLASS2.csv")
train_classes_1 = pd.read_csv("IMNT_DATASET_CLASS2.csv")
train_1 = train_1.drop(columns=['Class', 'Issues'])
train_classes_1 = train_classes_1.drop(columns=['Tasks','Bugs','Issues'])

# Devide data to test and train data with ratio 20 : 80
test_rows_to_select = int(0.2 * len(train))
train_rows_to_select= int(0.8 * len(train))

d1_x1 = train.drop(columns=['Bugs']).to_numpy()[:test_rows_to_select]   
d1_x2 = train.drop(columns=['Tasks']).to_numpy()[:test_rows_to_select]


d2_x1 = train_1.drop(columns=['Bugs']).to_numpy()[:test_rows_to_select]   
d2_x2 = train_1.drop(columns=['Tasks']).to_numpy()[:test_rows_to_select]


dati = np.vstack((train.to_numpy()[:train_rows_to_select], train_1.to_numpy()[:train_rows_to_select]))

# Added -1 cuz it should always predict between 0 and 1 !!!
classes = np.vstack((train_classes[:train_rows_to_select]-1, train_classes_1[:train_rows_to_select]-1)).flatten().tolist()

perc = Perceptrons(2)

for epoch in range(1000):
    for di in range(len(dati)):
        perc.fit(dati[di], classes[di])
        
img = np.zeros((211, 211))

for yi in range(211):
    for xi in range(211):
        img[yi][xi] = perc.predict([xi/10, yi/10])

plt.scatter(d1_x1, d1_x2, s=70, c='green', marker='o', linewidths=1, edgecolors='black', label='Class 1 (Medium Sprint)')
plt.scatter(d2_x1, d2_x2, s=70, c='blue', marker='o', linewidths=1, edgecolors='black', label='Class 2 (Hard Sprint)')
plt.imshow(img, extent=[0,21,0,21], origin='lower')

plt.title('Class 1 vs Class 2 clusterization')
plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))

## THIS IS DONE BY ChatGPT
# Initialize variables to store error metrics
train_rmse = 0
test_rmse = 0
train_misclassified = 0
test_misclassified = 0

# Calculate error metrics for training data
for di in range(len(dati)):
    predicted_class = perc.predict(dati[di])
    if classes[di] == 0:
        train_rmse += (predicted_class - 0) ** 2
        if round(predicted_class) != 0:
            train_misclassified += 1
    elif classes[di] == 1:
        train_rmse += (predicted_class - 1) ** 2
        if round(predicted_class) != 1:
            train_misclassified += 1

train_rmse = np.sqrt(train_rmse / len(dati))

# Calculate error metrics for test data
for di in range(len(d1_x1)):
    predicted_class = perc.predict([d1_x1[di][0], d1_x2[di][0]])
    test_rmse += (predicted_class - 0) ** 2
    if round(predicted_class) != 0:
        test_misclassified += 1

for di in range(len(d2_x1)):
    predicted_class = perc.predict([d2_x1[di][0], d2_x2[di][0]])
    test_rmse += (predicted_class - 1) ** 2
    if round(predicted_class) != 1:
        test_misclassified += 1

test_rmse = np.sqrt(test_rmse / (len(d1_x1) + len(d2_x1)))
print("CLASS 1 vs CLASS 2")
print("Training RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
print("Training Misclassified Count:", train_misclassified)
print("Test Misclassified Count:", test_misclassified)



plt.show()

#################################################################################################################################

train = pd.read_csv("IMNT_DATASET_CLASS0.csv")
train_classes = pd.read_csv("IMNT_DATASET_CLASS0.csv")
train = train.drop(columns=['Class', 'Issues'])
train_classes = train_classes.drop(columns=['Tasks','Bugs','Issues'])

train_1 = pd.read_csv("IMNT_DATASET_CLASS2.csv")
train_classes_1 = pd.read_csv("IMNT_DATASET_CLASS2.csv")
train_1 = train_1.drop(columns=['Class', 'Issues'])
train_classes_1 = train_classes_1.drop(columns=['Tasks','Bugs','Issues'])

test_rows_to_select = int(0.2 * len(train))
train_rows_to_select= int(0.8 * len(train))

d1_x1 = train.drop(columns=['Bugs']).to_numpy()[:test_rows_to_select]   
d1_x2 = train.drop(columns=['Tasks']).to_numpy()[:test_rows_to_select]


d2_x1 = train_1.drop(columns=['Bugs']).to_numpy()[:test_rows_to_select]   
d2_x2 = train_1.drop(columns=['Tasks']).to_numpy()[:test_rows_to_select]

# It is necessarry to slow down the learning rate because the gap between values is too big
lr = 0.01

dati = np.vstack((train.to_numpy()[:train_rows_to_select], train_1.to_numpy()[:train_rows_to_select]))
classes = np.vstack((train_classes[:train_rows_to_select], train_classes_1[:train_rows_to_select])).flatten().tolist()

perc = Perceptrons(2)

for epoch in range(1000):
    for di in range(len(dati)):
        perc.fit(dati[di], classes[di])
        
img = np.zeros((211, 211))

for yi in range(211):
    for xi in range(211):
        img[yi][xi] = perc.predict([xi/10, yi/10])

plt.scatter(d1_x1, d1_x2, s=70, c='green', marker='o', linewidths=1, edgecolors='black', label='Class 0 (Easy Sprint)')
plt.scatter(d2_x1, d2_x2, s=70, c='blue', marker='o', linewidths=1, edgecolors='black', label='Class 2 (Hard Sprint)')
plt.imshow(img, extent=[0,21,0,21], origin='lower')

plt.title('Class 0 vs Class 2 clusterization')
plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))

## THIS IS DONE BY ChatGPT
# Initialize variables to store error metrics
train_rmse = 0
test_rmse = 0
train_misclassified = 0
test_misclassified = 0

# Calculate error metrics for training data
for di in range(len(dati)):
    predicted_class = perc.predict(dati[di])
    if classes[di] == 0:
        train_rmse += (predicted_class - 0) ** 2
        if round(predicted_class) != 0:
            train_misclassified += 1
    elif classes[di] == 1:
        train_rmse += (predicted_class - 1) ** 2
        if round(predicted_class) != 1:
            train_misclassified += 1

train_rmse = np.sqrt(train_rmse / len(dati))

# Calculate error metrics for test data
for di in range(len(d1_x1)):
    predicted_class = perc.predict([d1_x1[di][0], d1_x2[di][0]])
    test_rmse += (predicted_class - 0) ** 2
    if round(predicted_class) != 0:
        test_misclassified += 1

for di in range(len(d2_x1)):
    predicted_class = perc.predict([d2_x1[di][0], d2_x2[di][0]])
    test_rmse += (predicted_class - 1) ** 2
    if round(predicted_class) != 1:
        test_misclassified += 1

test_rmse = np.sqrt(test_rmse / (len(d1_x1) + len(d2_x1)))
print("CLASS 0 vs CLASS 2")
print("Training RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
print("Training Misclassified Count:", train_misclassified)
print("Test Misclassified Count:", test_misclassified)



plt.show()
