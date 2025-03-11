"""
FSR-MSAGRU(load_model).py: Predicting PM2.5 using FSR-MSAGRU model
"""
import numpy as np
import pandas as pd
import keras.backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, RepeatVector, TimeDistributed, GlobalAveragePooling1D, Input
from keras import optimizers
from math import sqrt
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

# Define input and output parameters
step_in = 10  # Input time steps
n_features_in = 4  # Number of input features
n_features_out = 1  # Number of output features
step_out = 1  # Output time step
m = 1  # Experiment index
modelname = 'model1.1793-0.00.h5'  # Pre-trained model filename

# Load dataset
# Step ①: Import data
df = pd.read_excel("data_split.xlsx", header=0, index_col=0, parse_dates=True)
data_step_test = pd.read_excel('data_step_test1.xlsx', header=0, index_col=0, parse_dates=True)

df.isnull().sum()  # Check for missing values
data_all_values = df.values.astype('float32')  # Convert to float32
n_rows = data_all_values.shape[0]

# Step ②: Data preprocessing
# ②.1 Normalize the data
data_all_values = data_all_values.reshape((n_rows * n_features_in, 1))  # Convert to a single column
scaler = StandardScaler()
data_scaled_all_values = scaler.fit_transform(data_all_values)
data_scaled_all_values = data_scaled_all_values.reshape((n_rows, n_features_in))

data_step_test_values = data_step_test.values

# Prepare input and output data for model
n_obs = step_in * n_features_in  # Total observation size for input sequence
test_X, test_Y = data_step_test_values[:, :n_obs], data_step_test_values[:, n_obs:]

# Reshape input into 3D format for model (samples, time steps, features)
test_X = test_X.reshape((test_X.shape[0], step_in, n_features_in))

# Step ③: Load and use the pre-trained model
model = load_model(modelname)
Y_pred = model.predict(test_X)

# Step ④: Reverse normalization

def invert_scaling(data, name):
    """Reverse normalization and reshape the data"""
    data = data.reshape((data.shape[0], step_out * n_features_out))
    inv_data = scaler.inverse_transform(data)
    names = [(name + '%d(t)' % (j + 1)) if i == 0 else (name + '%d(t+%d)' % (j + 1, i))
             for i in range(step_out) for j in range(n_features_out)]
    return DataFrame(inv_data, columns=names)

inv_Y_pred = invert_scaling(Y_pred, 'Predicted')  # Reverse normalization for predictions
inv_Y_true = invert_scaling(test_Y, 'Actual')  # Reverse normalization for actual values

# Save results to an Excel file
result_data = concat([inv_Y_true, inv_Y_pred], axis=1)
result_data.to_excel('result_data(load_model)%d.xlsx' % m)

# Convert DataFrame to numpy arrays
inv_Y_true, inv_Y_pred = inv_Y_true.values, inv_Y_pred.values

# Step ⑤: Evaluation Metrics

def direction_accuracy(actual, predicted):
    """Calculate the direction accuracy of predicted values"""
    actual_changes = np.diff(actual)
    predicted_changes = np.diff(predicted)
    correct_direction = np.sum(np.sign(actual_changes) == np.sign(predicted_changes))
    return correct_direction / len(actual_changes)


def smape(actual, predicted):
    """Calculate the Symmetric Mean Absolute Percentage Error (SMAPE)"""
    denominator = np.abs(actual) + np.abs(predicted)
    valid_mask = denominator != 0
    smape_values = np.abs(actual - predicted) / (denominator / 2.0)
    smape_values[~valid_mask] = 0  # Handle zero denominator case
    return np.mean(smape_values[valid_mask])

# Compute evaluation metrics
mse = mean_squared_error(inv_Y_true, inv_Y_pred)  # Mean Squared Error
rmse = sqrt(mse)  # Root Mean Squared Error
mae = mean_absolute_error(inv_Y_true, inv_Y_pred)  # Mean Absolute Error
mape = mean_absolute_percentage_error(inv_Y_true, inv_Y_pred)  # Mean Absolute Percentage Error
smape = smape(inv_Y_true, inv_Y_pred)  # SMAPE
pcc = np.corrcoef(inv_Y_true, inv_Y_pred)[0, 1]  # Pearson Correlation Coefficient
mbe = np.mean(inv_Y_pred - inv_Y_true)  # Mean Bias Error (MBE)
mad = np.median(np.abs(inv_Y_pred - inv_Y_true))  # Median Absolute Deviation (MAD)
da = direction_accuracy(inv_Y_true, inv_Y_pred)  # Direction Accuracy

# Store evaluation results
Evaluation_index_dict = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'SMAPE': smape,
                          'PCC': pcc, 'MBE': mbe, 'MAD': mad, 'DA': da}
Evaluation_index = DataFrame(Evaluation_index_dict, index=[1])
print(Evaluation_index)

# Save evaluation metrics to an Excel file
Evaluation_index.to_excel('Evaluation_index(load_model)%d.xlsx' % m, index_label="Experiment Index")
