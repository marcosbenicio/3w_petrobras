import numpy as np
import pandas as pd


def compute_fft(df_instance, column):
    start_t = df_instance.index.min()
    end_t = df_instance.index.max()
    time_series = df_instance[f'{column}']

    fft_data = np.fft.fft(time_series, n=len(time_series))

    frequencies = np.fft.fftfreq(len(time_series), (end_t - start_t).total_seconds() / 60)
    return frequencies, fft_data

def smooth_fft_components(frequencies, fft_data, window_size=20):

    real = np.real(fft_data)
    imaginary = np.imag(fft_data)
    phase = np.angle(fft_data)
    magnitude= np.abs(fft_data)
    df_fft = pd.DataFrame({'frequency': frequencies, 'real': real, 'imaginary': imaginary, 'magnitude': magnitude, 'phase': phase})
    
    # Apply a rolling mean to smooth the imaginary part
    df_fft['real'] = df_fft['real'].rolling(window=window_size, center=True).mean()
    df_fft['imaginary'] = df_fft['imaginary'].rolling(window=window_size, center=True).mean()
    df_fft['magnitude'] = df_fft['magnitude'].rolling(window=window_size, center=True).mean()
    df_fft['phase'] = df_fft['phase'].rolling(window=window_size, center=True).mean()

    df_fft['window_size'] = window_size

    df_fft.dropna(inplace=True)
    
    return df_fft

def get_fft_to_ids(df, columns, ids=None, window_size=20):
    fft_results = []  
    
    # Ensure the index is a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    # If ids_to_process is provided, filter the DataFrame
    if ids is not None:
        df = df[df['id'].isin(ids)]

    # Group the DataFrame by 'id'
    grouped = df.groupby('id')

    for id_value, group in grouped:
        # Initialize a dictionary to hold FFT data for all features for this id
        fft_data_dict = {'id': id_value}
        if 'Instance' in group.columns:
            fft_data_dict['Instance'] = group['Instance'].iloc[0]
            fft_data_dict['Set'] = group['Set'].iloc[0]
            fft_data_dict['Class'] = group['Class'].iloc[0]

        # Loop over each feature
        for feature in columns:
            if feature in group.columns:
                # Apply the FFT function
                frequencies, fft_data = compute_fft(group, feature)
                # Apply smoothing to the FFT data
                df_fft = smooth_fft_components(frequencies, fft_data, window_size=window_size)

                # Rename columns to include feature name
                df_fft.rename(columns={
                    'frequency': f'{feature}_frequency',
                    'real': f'{feature}_real',
                    'imaginary': f'{feature}_imaginary',
                    'magnitude': f'{feature}_magnitude',
                    'phase': f'{feature}_phase'
                }, inplace=True)

                # Convert df_fft to dictionary and update fft_data_dict
                for col in df_fft.columns:
                    fft_data_dict[col] = df_fft[col].values

        # Convert fft_data_dict to DataFrame
        df_fft_id = pd.DataFrame(fft_data_dict)
        fft_results.append(df_fft_id)
    df_fft_all = pd.concat(fft_results, ignore_index=True)
    return df_fft_all