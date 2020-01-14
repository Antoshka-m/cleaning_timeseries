from get_filenames import get_filenames
from scipy.stats import linregress
from scipy.stats import median_absolute_deviation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import os.path

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


def plot_sig_flattening(t, x, x_fit, x_flat, y_cut=None, title=None):
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
    if y_cut is not None:
        plt.plot([t.loc[0], t.loc[len(x_flat)-1]], [y_cut[0], y_cut[0]], 'r--')
        plt.plot([t.loc[0], t.loc[len(x_flat)-1]], [y_cut[1], y_cut[1]], 'r--')
    plt.xlabel('t, s', FontSize=18)
    plt.ylabel('Deflection', FontSize=16)
    plt.xticks(FontSize=16)
    plt.yticks(FontSize=16)
    if title is not None:
        plt.title(title)
    plt.grid()
    plt.show()
    
def plot_signal(t, x, title=None):
    
    plt.figure()
    if title is not None:
        plt.title(title)
    plt.plot(t, x)
    plt.grid()
    plt.xlabel('t, s', FontSize=16)
    plt.ylabel('Deflection', FontSize=16)
    plt.xticks(FontSize=16)
    plt.yticks(FontSize=16)
    plt.show()
    
def detect_peaks_one_mean(x, n=3):
    mean = np.mean(x)
    std = np.std(x)
    x_clean = np.copy(x)
    # df = pd.DataFrame({'x': x, 't': t})
    # x_clean = df[abs(df['x'])>abs(mean+2*std)]['x'].apply(
            # lambda x: np.random.uniform(mean-2*std, mean+2*std))
    # mask = (df['x'])
    # x_clean = df['x'].apply(lambda x: 'np.random.uniform(mean-2*std, mean+2*std)' if abs(x)>abs(mean+2*std))
    x_clean[abs(x_clean)>abs(mean+n*std)]=np.random.uniform(mean-n*std, mean+n*std, len(x_clean[abs(x_clean)>abs(mean+n*std)]))
    return x_clean, (mean-n*std, mean+n*std)


def detect_peaks_one_median(x, n=3):
    median = np.median(x)
    mad = median_absolute_deviation(x)
    x_clean = np.copy(x)
    x_clean[abs(x_clean)>abs(median+n*mad)]=np.random.uniform(median-n*mad, median+n*mad, len(x_clean[abs(x_clean)>abs(median+n*mad)]))
    return x_clean, (median-n*mad, median+n*mad)


def get_fps_from_fname(file):
    """Get fps from filename with stamp '_1000fps', where 1000 is fps value.
        
        Parameters
        ----------
        file : str 
            filename with full path
        
        Returns:
        -------
        fps: int
            fps of videofile (frames/second) """
            
    filename = os.path.basename(file)
    i = filename.find('fps')
    j=i
    while filename[j] != '_':
        j = j-1
    fps=int(filename[j+1:i])
    return fps

def get_export_dir(file):
    """
    create export directory (if it doesn't exist yet) in the same folder as import folder
        
    Returns
    -------
    exp_dir_path : str
        path of export folder
    """
    # exp_dir_name = "export_files"
    dir_name ='cleaned_data'
    exp_dir_path = os.path.join(os.path.dirname(file), dir_name) # export directory
    if not os.path.isdir(exp_dir_path): #create directory if it doesnt exist yet
        os.makedirs(exp_dir_path)
        print("\nExport directory %s was created. Export csvs will be saved there" % exp_dir_path)
    else:
        print("\nExport directory %s already exists. Export csvs will be saved there" % exp_dir_path)
    return exp_dir_path


def export_cleaned_data(data, path, in_fname, out_fname, fps):
    data=pd.concat([data, pd.DataFrame({'fps': [fps]})], axis=1)
    output_name = in_fname + '_'+ out_fname+'.csv'
    data.to_csv(os.path.join(path, output_name))
    print('Exported %s file' %output_name)

def export_fig(path, in_fname, out_fname):
    output_name = in_fname + '_'+ out_fname+'.png'
    plt.savefig(output_name)
file_list = get_filenames()
for file in file_list:
    df = pd.read_csv(file)
    fps=get_fps_from_fname(file)
    print('Read file %s recorded with %s fps...' %(os.path.basename(file), fps))
    y = df['Iz']
    x = df['Il']
    y_norm = df['Iz norm']
    x_norm = df['Il norm']
    t= df['t']
    dt = df['t'].loc[1]-df['t'].loc[0]
    x_fit, x_flat = flattening(data=x, dt=dt, chunk_l=60)
    x_norm_fit, x_norm_flat = flattening(data=x_norm, dt=dt, chunk_l=60)
    y_fit, y_flat = flattening(data=y, dt=dt, chunk_l=60)
    y_norm_fit, y_norm_flat = flattening(data=y_norm, dt=dt, chunk_l=60)
    x_sd_cleaned, x_cut_sd = detect_peaks_one_mean(x_flat, n=3)
    y_sd_cleaned, y_cut_sd = detect_peaks_one_mean(y_flat, n=3)
    x_norm_sd_cleaned, x_norm_cut_sd = detect_peaks_one_mean(x_norm_flat, n=3)
    y_norm_sd_cleaned, y_norm_cut_sd = detect_peaks_one_mean(y_norm_flat, n=3)
    x_mad_cleaned, x_cut_mad = detect_peaks_one_median(x_flat, n=3)
    y_mad_cleaned, y_cut_mad = detect_peaks_one_median(y_flat, n=3)
    x_norm_mad_cleaned, x_norm_cut_mad = detect_peaks_one_median(x_norm_flat, n=3)
    y_norm_mad_cleaned, y_norm_cut_mad = detect_peaks_one_median(y_norm_flat, n=3)
    x_medfilt = medfilt(x_flat, kernel_size=5)
    y_medfilt = medfilt(y_flat, kernel_size=5)
    x_norm_medfilt = medfilt(x_norm_flat, kernel_size=5)
    y_norm_medfilt = medfilt(y_norm_flat, kernel_size=5)
    df_flat = pd.DataFrame({'t': t.loc[:len(x_flat)-1], 
                                             'Il': x_flat,
                                             'Iz': y_flat,
                                             'Il norm': x_norm_flat,
                                             'Iz norm': y_norm_flat})
    df_sd_cleaned = pd.DataFrame({'t': t.loc[:len(x_flat)-1], 
                                             'Il': x_sd_cleaned,
                                             'Iz': y_sd_cleaned,
                                             'Il norm': x_norm_sd_cleaned,
                                             'Iz norm': y_norm_sd_cleaned})
    df_mad_cleaned = pd.DataFrame({'t': t.loc[:len(x_flat)-1], 
                                             'Il': x_mad_cleaned,
                                             'Iz': y_mad_cleaned,
                                             'Il norm': x_norm_mad_cleaned,
                                             'Iz norm': y_norm_mad_cleaned})
    df_med_filtered = pd.DataFrame({'t': t.loc[:len(x_flat)-1], 
                                             'Il': x_medfilt,
                                             'Iz': y_medfilt,
                                             'Il norm': x_norm_medfilt,
                                             'Iz norm': y_norm_medfilt})
    print('Going to export calculated data...')
    in_filename = os.path.basename(file)[:-4]
    export_dir = get_export_dir(file)
    print('Exporting flattened data...')
    export_cleaned_data(data=df_flat,
                        path=export_dir,
                        in_fname=in_filename,
                        out_fname='flat',
                        fps=fps)
    export_cleaned_data(data=df_sd_cleaned,
                        path=export_dir,
                        in_fname=in_filename,
                        out_fname='sd_cleaned',
                        fps=fps)
    export_cleaned_data(data=df_mad_cleaned,
                        path=export_dir,
                        in_fname=in_filename,
                        out_fname='mad_cleaned',
                        fps=fps)
    export_cleaned_data(data=df_med_filtered,
                        path=export_dir,
                        in_fname=in_filename,
                        out_fname='med_filtered',
                        fps=fps)
    print('Plotting flattening of horizontal axis (Il), shown SD cut')
    plot_sig_flattening(t, x, x_fit, x_flat, x_cut_sd, title='Flattening Il')
    print('Plotting signal after peaks removal, Il')
    plot_signal(t.loc[:len(x_flat)-1], x_sd_cleaned, title='Removed peaks outside 3 SD, Il')
    print('Plotting flattening of vertical axis (Iz), shown SD cut')
    plot_sig_flattening(t, y, y_fit, y_flat, y_cut_sd, title='Flattening Iz')
    print('Plotting signal after peaks removal, Iz')
    plot_signal(t.loc[:len(x_flat)-1], y_sd_cleaned, title='Removed peaks outside of 3 SD, Iz')
    # now do removal of peak using median
    print('Plotting flattening of horizontal axis (Il), shown MAD cut')
    plot_sig_flattening(t, x, x_fit, x_flat, x_cut_mad, title='Flattening Il')
    print('Plotting signal after mad peaks removal, Il')
    plot_signal(t.loc[:len(x_flat)-1], x_mad_cleaned, title='Removed peaks outside 3 MAD, Il')
    print('Plotting flattening of vertical axis (Iz), shown 3 MAD cut')
    plot_sig_flattening(t, y, y_fit, y_flat, y_cut_mad, title='Flattening Iz')
    print('Plotting signal after peaks removal, Iz')
    plot_signal(t.loc[:len(x_flat)-1], y_mad_cleaned, title='Removed peaks outside of 3 MAD, Iz')
    print('Plotting median filtered signal, Il')
    plot_signal(t.loc[:len(x_flat)-1], x_medfilt, title='Median filtered Il')
    print('Plotting median filtered signal, Iz')
    plot_signal(t.loc[:len(x_flat)-1], y_medfilt, title='Median filtered Iz')
