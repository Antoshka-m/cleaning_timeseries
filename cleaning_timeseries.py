from get_filenames import get_filenames
from scipy.stats import linregress
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def flattening(data, dt, chunk_l):
    """
    Make flattening of raw data using windows (time chunks) of given size
    
    Parameters
    -------
    
    data: list or np.array
        timeseries data (deflection magnitude)
    dt: float
        time resolution of data
    chunk_l: float
        length of the chunk (in seconds)
        
    Returns
    -------
    data_fit: np.array
        fitting line
    data_flat: np.array
        flattened data
    """
    # get chunk length in datapoints instead of seconds
    chunk_l = int(chunk_l//dt )
    # get size of output flattened data (cut out remaining part after division)
    flat_data_size = len(data)//chunk_l
    data_fit = np.empty((flat_data_size, chunk_l))
    for chunk_i in range(flat_data_size):
        start_i = chunk_i*chunk_l
        end_i = (chunk_i+1)*chunk_l
        y = data[start_i:end_i]
        # make linear regression for each chunk
        k_coef, b_coef, _, _, _ = linregress([i for i in range(len(y))], y)
        yfit = k_coef * np.array([i for i in range(len(y))]) + b_coef
        data_fit[chunk_i] = yfit
    data_flat = data[:(flat_data_size*chunk_l)]-data_fit.flatten()
    return data_fit, data_flat


def plot_sig_flattening(t, x, x_fit, x_flat):
    """
    Plotting raw signal, linear fits, and flattened signal.
    
    Parameters
    --------
    
    t: np-array or Series:
        time
    x: np-array or Series:
        deflection (raw timeseries data)
    x_fit: np-array:
        x values of fit
    x_flat: np-array:
        flattened data (after substraction of fit from raw data)
    """
    #plt.figure(figsize=(15,10))
    plt.figure()
    plt.plot(t, x)
    plt.xlabel('t, s', FontSize=16)
    plt.ylabel('Deflection', FontSize=16)
    plt.xticks(FontSize=16)
    plt.yticks(FontSize=16)
    plt.title('Raw signal')
    plt.grid()
    for i, fit in enumerate(x_fit):
        plt.plot(t.loc[i*len(x_fit[0]):(i+1)*len(x_fit[0])-1],
                       x_fit[i],
                       color='purple')
    plt.figure()
    plt.plot(t.loc[:len(x_flat)-1], x_flat)
    plt.xlabel('t, s', FontSize=18)
    plt.ylabel('Deflection', FontSize=16)
    plt.xticks(FontSize=16)
    plt.yticks(FontSize=16)
    plt.title('Flattened signal')
    plt.grid()
    plt.show()

file_list = get_filenames()
for file in file_list:
    df = pd.read_csv(file)
    y = df['Iz']
    x = df['Il']
    t= df['t']
    dt = df['t'].loc[1]-df['t'].loc[0]
    x_fit, x_flat = flattening(data=x, dt=dt, chunk_l=30)
    y_fit, y_flat = flattening(data=y, dt=dt, chunk_l=30)
    print('Plotting flattening of horizontal axis (Il)')
    plot_sig_flattening(t, x, x_fit, x_flat)
    print('Plotting flattening of vertical axis (Iz)')
    plot_sig_flattening(t, y, y_fit, y_flat)