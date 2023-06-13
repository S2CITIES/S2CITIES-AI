"""
This file trains a model on the extracted features using neural networks.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras.utils import to_categorical
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.python.client import device_lib

from modelevaluator import ModelEvaluator

# Set working directory to this file's directory using pathlib
os.chdir(Path(__file__).parent)

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

gpus = get_available_gpus()

# Check that we are using GPU
print(f'GPU available: {gpus}')

# If not, exit
if not gpus:
    print('No GPU available, exiting.')
    exit()

# Get the number of samples ignoring hidden files
num_samples = len([f for f in os.listdir('dataset_creation/5_features_extracted/0') if not f.startswith('.')]) + len([f for f in os.listdir('dataset_creation/5_features_extracted/1') if not f.startswith('.')])

# Get the number of classes ignoring hidden files
num_classes = len([f for f in os.listdir('dataset_creation/5_features_extracted') if not f.startswith('.')])

# Get number of time steps and features from the first file
num_timesteps, num_features = np.load(os.path.join('dataset_creation/5_features_extracted/0', os.listdir('dataset_creation/5_features_extracted/0')[0])).shape

list_0 = []
list_1 = []
y = []

# Read npy files and append to lists
for file in os.listdir('dataset_creation/5_features_extracted/0'):
    if file.endswith('.npy'):
        data = np.load(os.path.join('dataset_creation/5_features_extracted/0', file))
        list_0.append(data)
        y.append(0)
        
for file in os.listdir('dataset_creation/5_features_extracted/1'):
    if file.endswith('.npy'):
        data = np.load(os.path.join('dataset_creation/5_features_extracted/1', file))
        list_1.append(data)
        y.append(1)

# X must be a 3D array of shape (samples, timesteps, features)
# y must be a 2D array of shape (samples, classes)
X_data = np.concatenate((list_0, list_1), axis=0)
y_data = np.array(y)
y_data = to_categorical(y_data).astype(int)

print(f'{X_data.shape=}')
print(f'{y_data.shape=}')
print(f'We have {X_data.shape[0]} samples, each of which is a time series with {X_data.shape[1]} time steps and {X_data.shape[2]} features. We have to classify {y_data.shape[1]} classes.')

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle=True, random_state=42, stratify=y_data)

# Create validation set from train set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, random_state=42, stratify=y_train)

# Create `logs` directory for TensorBoard
LOG_DIR = Path() / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Build model using the functional API
inputs = keras.Input(shape=(num_timesteps, num_features))
x = LSTM(64, return_sequences=True, activation='relu', input_shape=(num_timesteps, num_features))(inputs)
x = LSTM(128, return_sequences=True, activation='relu')(x)
x = LSTM(64, return_sequences=False, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
outputs = Dense(num_classes, activation='softmax')(x)

# Build model
model = keras.Model(inputs=inputs, outputs=outputs, name='TomCruise')

# Compile model
model.compile(
    optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy'])

# Print model summary
print(model.summary())

# Create callbacks
callbacks=[
    TensorBoard(log_dir=LOG_DIR),
    EarlyStopping(monitor='val_categorical_accuracy', mode='max', patience=15, restore_best_weights=True)
]

# Train model
model.fit(
    X_train, y_train,
    epochs=2000,
    callbacks=callbacks,
    validation_data=(X_val, y_val),
    batch_size=16
)

# Save model
MODELS_DIR = Path() / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
model.save(MODELS_DIR / 'baseline.h5')
model = keras.models.load_model(MODELS_DIR / 'baseline.h5')

# Evaluate model
y_test_proba = model.predict(X_test)[:,1]

evaluator = ModelEvaluator(model_name='LSTM NN', y_true=y_test[:,1], y_proba=y_test_proba, threshold=0.5)
evaluator.evaluate_metrics()
evaluator.plot_roc_curve()
evaluator.plot_confusion_matrix()
