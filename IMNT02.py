import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import random

# Read the dataset from 'IMNT_DATASET.csv'
dati_df = pd.read_csv('IMNT_DATASET.csv')

# Extract the target variable and one-hot encode it
gaidamas_vertibas = pd.get_dummies(dati_df['Class'], dtype=float)

# Split the dataset into training and testing sets
ieejas_train, ieejas_test, gaidamas_train, gaidamas_test = train_test_split(
    dati_df.drop(columns=['Class']),  # Remove the 'Class' column
    gaidamas_vertibas,
    test_size=0.3,
    stratify=gaidamas_vertibas
)

# Calculate means and standard deviations for normalization
means = ieejas_train.mean()
stds = ieejas_train.std()

# Normalize the input features
ieejas_train = (ieejas_train - means) / stds
ieejas_test = (ieejas_test - means) / stds

# Convert data to NumPy arrays
ieejas_train = ieejas_train.to_numpy()
ieejas_test = ieejas_test.to_numpy()
gaidamas_train = gaidamas_train.to_numpy()
gaidamas_test = gaidamas_test.to_numpy()

# Set random seeds for reproducibility
SEED = 123
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Initialize the model
initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1., seed=SEED)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3, activation='linear', kernel_initializer=initializer, bias_initializer=initializer, input_shape=(ieejas_train.shape[1],)),
    tf.keras.layers.Dense(2, activation='linear', kernel_initializer=initializer, bias_initializer=initializer),
    tf.keras.layers.Dense(gaidamas_vertibas.shape[1], activation='softmax', kernel_initializer=initializer, bias_initializer=initializer)
])

# Compile the model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(ieejas_train, gaidamas_train, epochs=250)

# Evaluate the model on the test s 
result = model.evaluate(ieejas_test, gaidamas_test)
print("Test loss:", result[0])
print("Test accuracy:", result[1])