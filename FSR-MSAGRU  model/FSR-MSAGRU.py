"""FSR-MSAGRU.py: Predicting PM2.5 using FSR-MSAGRU model"""

import numpy as np
import pandas as pd
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling1D, MultiHeadAttention
from keras.layers import GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from math import ceil, sqrt
from matplotlib import pyplot as plt
from pandas import concat, DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

# Define data split ratios
split = 0.8  # Train-validation-test split ratio
split2 = 0.5  # Validation-test split ratio

# Define time-series parameters
step_in = 10  # Input time steps for multi-step forecasting
step_out = 1  # Output time steps
n_features_in = 4  # Number of input features
n_features_out = 1  # Number of output features
n = 5  # Number of model iterations
num_heads = 4  # Number of attention heads

# Load dataset
df = pd.read_excel("data_split.xlsx", header=0, index_col=0, parse_dates=True)
df.isnull().sum()  # Check for missing values

# Convert dataframe to numerical array
data_all_values = df.values.astype('float32')
n_rows = data_all_values.shape[0]
n_features_in = data_all_values.shape[1]

# Normalize data
scaler = StandardScaler()
data_scaled_all_values = scaler.fit_transform(data_all_values)

# Split dataset into train, validation, and test sets
data_split_1 = ceil(len(data_scaled_all_values) * split)
data_train = data_scaled_all_values[:data_split_1, :]
data_temp = data_scaled_all_values[data_split_1:, :]
data_split_2 = ceil(len(data_temp) * split2)
data_valid = data_temp[:data_split_2, :]
data_test = data_temp[data_split_2:, :]


# Convert time-series data to supervised learning format
def series_to_supervised(data, step_in, step_out, dropnan=True):
    """
    Convert time series data into supervised learning format.
    Args:
        data: Input time-series data
        step_in: Number of past time steps to use as input
        step_out: Number of future time steps to predict
        dropnan: Whether to drop NaN values
    Returns:
        Transformed DataFrame
    """
    data_value = DataFrame(data)
    cols, names = list(), list()

    # Create input sequence (lag features)
    for i in range(step_in):
        cols.append(data_value.shift(-i))
        names += [('var%d(t%d)' % (j + 1, i - step_in)) for j in range(n_features_in)]

    # Create output sequence (forecast horizon)
    for i in range(step_out):
        cols.append(data_value.shift(-(i + step_in)))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_features_in)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_features_in)]

    # Concatenate input and output sequences
    agg = concat(cols, axis=1)
    agg.columns = names

    # Remove NaN values
    if dropnan:
        agg.dropna(inplace=True)

    # Drop unnecessary columns
    def data_step_drop(data1):
        for k in range(step_out):
            data1.drop(data1.columns[range(len(data1.columns) - n_features_in + n_features_out - k * n_features_out,
                                           len(data1.columns) - k * n_features_out, 1)], axis=1, inplace=True)
        return data1

    if n_features_out != n_features_in:
        agg = data_step_drop(agg)
    return agg


# Prepare data for training
train_data = series_to_supervised(data_train, step_in, step_out)
valid_data = series_to_supervised(data_valid, step_in, step_out)
test_data = series_to_supervised(data_test, step_in, step_out)

# Convert data into numpy arrays
train_X, train_Y = train_data.values[:, :-n_features_out], train_data.values[:, -n_features_out:]
valid_X, valid_Y = valid_data.values[:, :-n_features_out], valid_data.values[:, -n_features_out:]
test_X, test_Y = test_data.values[:, :-n_features_out], test_data.values[:, -n_features_out:]

# Reshape input data for  models
train_X = train_X.reshape((train_X.shape[0], step_in, n_features_in))
valid_X = valid_X.reshape((valid_X.shape[0], step_in, n_features_in))
test_X = test_X.reshape((test_X.shape[0], step_in, n_features_in))

# Build and train multiple models
for h in range(n):
    # Define input layer
    input_layer = Input(shape=(train_X.shape[1], train_X.shape[2]))

    # GRU layer
    gru_layer = GRU(128, return_sequences=True, activation='relu')(input_layer)

    # Multi-head attention layer
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=n_features_in // num_heads)(gru_layer, gru_layer)

    # Global average pooling layer
    pooled_output = GlobalAveragePooling1D()(attention_output)

    # Output layer
    output_layer = Dense(step_out * n_features_out)(pooled_output)

    # Compile model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Define callbacks
    csv_logger = CSVLogger(f'training{h + 1}.log')
    checkpoint = ModelCheckpoint(filepath=f'model{h + 1}.h5', monitor='val_loss', save_best_only=True, mode='auto')
    callbacks = [EarlyStopping(monitor='val_loss', patience=500)]

    # Train model
    history = model.fit(train_X, train_Y, epochs=20000, batch_size=64, callbacks=[checkpoint, csv_logger, callbacks],
                        validation_data=(valid_X, valid_Y), verbose=2, shuffle=False)

    # Plot training loss
    plt.plot(history.history['loss'], 'r', label='train_loss')
    plt.plot(history.history['val_loss'], 'b', label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Make predictions
    Y_pred = model.predict(test_X)

    # Evaluate model performance
    mse = mean_squared_error(test_Y, Y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(test_Y, Y_pred)
    mape = mean_absolute_percentage_error(test_Y, Y_pred)


    # Save evaluation metrics
    metrics_df = pd.DataFrame({'MSE': [mse], 'RMSE': [rmse], 'MAE': [mae], 'MAPE': [mape]})
    metrics_df.to_excel(f'Evaluation_index{h + 1}.xlsx', index=False)

