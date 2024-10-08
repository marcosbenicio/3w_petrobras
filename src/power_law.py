import numpy as np
import pandas as pd
from scipy.stats import linregress

def fit_powerlaw_to_features(df, features):
    results = []
    
    # Group by 'id' and 'Instance' to consider each unique time series
    for (id, instance), group in df.groupby(['id', 'Instance']):
        for feature in features:
            freq_col = f'{feature}_frequency'
            mag_col = f'{feature}_magnitude'
            
            if freq_col in group.columns and mag_col in group.columns:
                frequencies = group[freq_col].values
                magnitudes = group[mag_col].values
                
                # Filter valid frequencies and magnitudes (non-zero, positive values)
                valid = (frequencies > 0) & (magnitudes > 0)
                if valid.any():
                    frequencies = frequencies[valid]
                    magnitudes = magnitudes[valid]
                    
                    log_freq = np.log10(frequencies)
                    log_mag = np.log10(magnitudes)
                    
                    # Perform linear regression in the log-log space
                    slope, intercept, r_value, p_value, std_err = linregress(log_freq, log_mag)
                    alpha = -slope
                    
                    result_dict = {
                        'id': id,
                        'Instance': instance,
                        'feature': feature,
                        'alpha': alpha,
                        'intercept': intercept,
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'std_err': std_err
                    }
                    results.append(result_dict)
    # Convert the list of dictionaries to a DataFrame
    return pd.DataFrame(results)