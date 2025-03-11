from statsmodels.tsa.seasonal import STL, seasonal_decompose
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# Set the figure size for plots
plt.rc("figure", figsize=(10, 6))

# Load the dataset
filename = 'daydatas(cityname).xlsx'
df = pd.read_excel(filename)

# Display dataset information
df.info()
print(df.isnull().sum())

city_name = 'cityname'


# Function to fill missing values using the nearest neighbor mean method
def knm(df1, n, dataname):
    """
    Fill missing values using the nearest neighbor mean method.

    Parameters:
        df1 (DataFrame): Input data frame containing missing values.
        n (int): Number of nearest neighbors used for filling missing values.
        dataname (str): Column name to process.

    Returns:
        DataFrame: The processed data frame with missing values filled.
    """
    temp = df1.isnull().T.any().values  # Identify missing value rows
    temp_df = df1.copy()

    for i in range(len(temp)):
        if temp[i] == True:
            if i < n - 1:  # If the missing value is within the first 'n' rows
                temp_df.loc[i, dataname] = df1.loc[i:i + n, dataname].mean()
            elif i > len(temp) - 1 - n:  # If the missing value is within the last 'n' rows
                temp_df.loc[i, dataname] = df1.loc[i - n:i, dataname]
            else:  # General case for missing values in the middle of the data
                temp_df.loc[i, dataname] = df1.loc[i - n:i + n, dataname].mean()

    return temp_df


# Function to analyze periodicity using Fourier Transform
def evaluation_period():
    """
    Analyze the periodicity of the time series using Fourier Transform.

    Returns:
        float: The estimated period of the sequence.
    """
    df3 = pd.read_excel(filename, header=0, index_col=0, parse_dates=True)
    y = df3.values

    # Perform Fourier Transform
    yf = np.fft.fft(y)

    # Compute the frequency spectrum
    freq = np.fft.fftfreq(len(y))
    amp = np.abs(yf)

    # Create a DataFrame to store frequency and amplitude values
    df_spectra = pd.DataFrame({'Frequency': freq[:len(freq) // 2], 'Amplitude': np.squeeze(amp[:len(amp) // 2])})
    df_spectra.to_excel('spectra_data(%s).xlsx' % city_name)  # Save the frequency spectrum to an Excel file

    # Identify the maximum peak (ignoring the DC component)
    max_pos = np.argmax(amp[1:len(amp) // 2]) + 1
    period = 1 / freq[max_pos] if freq[max_pos] != 0 else None

    # Plot the original time series
    plt.subplot(2, 1, 1)
    plt.plot(y)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Original Sequence')

    # Plot the frequency spectrum
    plt.subplot(2, 1, 2)
    plt.plot(freq[:len(freq) // 2], amp[:len(amp) // 2])
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title('Periodicity Analysis')

    plt.tight_layout()
    plt.savefig('period1.png')

    print('The estimated period of the sequence is:', period)
    return period


# Apply nearest neighbor mean filling for missing values
not_miss = knm(df[[city_name]], 3, city_name)
df[city_name] = not_miss.values

# Display dataset information after filling missing values
df.info()
print(df.isnull().sum())

# Extract time series data
data = df[city_name]

# Perform periodicity analysis
period = evaluation_period()

# Perform seasonal decomposition using the estimated period
seasonal_decomp = seasonal_decompose(data, model="additive", period=round(period))

# Store decomposition results
df['trend'] = seasonal_decomp.trend
df['seasonal'] = seasonal_decomp.seasonal
df['resid'] = seasonal_decomp.resid

# Save the decomposed data to an Excel file
df.to_excel('data_split1.xlsx', index_label=str(period))

# Plot seasonal decomposition results
seasonal_decomp.plot()
plt.savefig('split1.png')
plt.show()
