from matplotlib import pyplot as plt
import numpy as np
from math import exp
import pandas as pd


w0 = 4.5  #4.5
w1 = 0.5  #0.5
w2 = 1  #1

def perc_out(x1, x2):
    z = -1*w0 + x1*w1 + x2*w2
    y = 1/(1+exp(-1*z))
    return y

def act_step(val):
    if val > 0:
        return 1
    else:
        return 0

def der_step(val):
    return 1

def act_sigmoid(val):
    return 1.0/(1.0 + exp(-val))

def der_sigmoid(val):
    s = act_sigmoid(val)
    return s * (1 - s)

lr = 0.1

class Perceptrons:
    def __init__(self, input_size: int):
        self.w = np.zeros(input_size)
        self.w0 = 0
        self.grad = 0
    
    def predict(self, input_values):  # input_values ir masivs ar tik vertibam cik svari, jeb input_size
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

# Devide data to test and train data with ratio 20 : 80
test_rows_to_select = int(0.2 * len(train))
train_rows_to_select= int(0.8 * len(train))

# Filling test data arrays. 
# Here I set 20% of all data to show it in data grid.
# It will not be used for training the perceptron!

# Fill test data for class 0 objects.
d1_x1 = train.drop(columns=['Bugs']).to_numpy()[:test_rows_to_select]   
d1_x2 = train.drop(columns=['Tasks']).to_numpy()[:test_rows_to_select]

# Fill test data for class 1 objects.
d2_x1 = train_1.drop(columns=['Bugs']).to_numpy()[:test_rows_to_select]   
d2_x2 = train_1.drop(columns=['Tasks']).to_numpy()[:test_rows_to_select]

# Fill TRAINING data for class 0 and class 1 objects. It is 80% of all class 0 and class 1 objects.
dati = np.vstack((train.to_numpy()[:train_rows_to_select], train_1.to_numpy()[:train_rows_to_select]))

# Define classes for objects. The amount of classes objects shoud be equal (31:31, 15:15 etc..).
classes = np.vstack((train_classes[:train_rows_to_select], train_classes_1[:train_rows_to_select])).flatten().tolist()

# Declare perceptron to sort Class 0 and Class 1 objects
perc = Perceptrons(2)

# Adjust weights for signals in 1000 epochs
for epoch in range(1000):
    for di in range(len(dati)):
        perc.fit(dati[di], classes[di])
        
img = np.zeros((121, 121))

# Calculate grid values (colors)
for yi in range(121):
    for xi in range(121):
        img[yi][xi] = perc.predict([xi/10, yi/10])

# Sketch test data on grid to validate perceptron accuaricy
plt.scatter(d1_x1, d1_x2, s=70, c='red', marker='o', linewidths=1, edgecolors='black', label='Class 0 (Easy Sprint)')
plt.scatter(d2_x1, d2_x2, s=70, c='green', marker='o', linewidths=1, edgecolors='black', label='Class 1 (Medium Sprint)')
plt.imshow(img, extent=[0,12,0,12], origin='lower')

plt.title('Class 0 vs Class 1 clusterization')
plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))


plt.show()

#################################################################################################################################

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

# Devide data to test and train data with ratio 20 : 80
test_rows_to_select = int(0.2 * len(train))
train_rows_to_select= int(0.8 * len(train))

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
        img[yi][xi] = perc.predict([xi/10, yi/10])

plt.scatter(d1_x1, d1_x2, s=70, c='red', marker='o', linewidths=1, edgecolors='black', label='Class 0 (Easy Sprint)')
plt.scatter(d2_x1, d2_x2, s=70, c='green', marker='o', linewidths=1, edgecolors='black', label='Class 1 (Medium Sprint)')
plt.imshow(img, extent=[0,12,0,12], origin='lower')

plt.title('Class 0 vs Class 1 clusterization')
plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))


plt.show()


d1_x1_1 = np.array([1,3,5])   #training the perceptron. d1 represents one class (class 0)
d1_x2_1 = np.array([1,1,1])   

d2_x1_1 = np.array([3,5,7])   #class 2
d2_x2_1 = np.array([3,3,3])

# dati_1 = [
#     [1, 1],
#     [3, 1],
#     [5, 1],
#     [3, 3],
#     [5, 3],
#     [7, 3]
# ]
# classes_1 = [
#     0,
#     0,
#     0,
#     1,
#     1,
#     1
# ]

# perc_1 = Perceptrons(2)

# for epoch in range(1000):
#     for di in range(len(dati_1)):
#         perc_1.fit(dati_1[di], classes_1[di])
        
# img_1 = np.zeros((51, 81))

# for yi in range(51):
#     for xi in range(81):
#         img_1[yi][xi] = perc_1.predict([xi/10, yi/10])

# plt.scatter(d1_x1_1, d1_x2_1, s=70, c='red', marker='o', linewidths=1, edgecolors='black')
# plt.scatter(d2_x1_1, d2_x2_1, s=70, c='green', marker='o', linewidths=1, edgecolors='black')
# plt.imshow(img_1, extent=[0,8,0,5], origin='lower')
# plt.show()


# d1_x1_2 = np.array([1,2,2])   #training the perceptron. d1 represents one class (class 0)
# d1_x2_2 = np.array([4,4,5])   

# d2_x1_2 = np.array([8,6,7])   #class 2
# d2_x2_2 = np.array([1,2,1])

# dati_2 = [
#     [1, 4],
#     [2, 4],
#     [2, 5],
#     [8, 1],
#     [6, 2],
#     [7, 1]
# ]
# classes_2 = [
#     0,
#     0,
#     0,
#     1,
#     1,
#     1
# ]

# perc_2 = Perceptrons(2)

# for epoch in range(10000):
#     for di in range(len(dati_2)):
#         perc_2.fit(dati_2[di], classes_2[di])
        
# img_2 = np.zeros((51, 81))

# for yi in range(51):
#     for xi in range(81):
#         img_2[yi][xi] = perc_2.predict([xi/10, yi/10])

# plt.scatter(d1_x1_2, d1_x2_2, s=70, c='red', marker='o', linewidths=1, edgecolors='black')
# plt.scatter(d2_x1_2, d2_x2_2, s=70, c='green', marker='o', linewidths=1, edgecolors='black')
# plt.imshow(img_2, extent=[0,8,0,5], origin='lower')
# plt.show()