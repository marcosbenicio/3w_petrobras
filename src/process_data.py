import pandas as pd
import numpy as np


def preprocess(df,instance, periods_undersample = None, positive_event = None):

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.index = df['timestamp']
    df = df.drop(columns=['timestamp'])
                

    df = df.ffill().bfill()

    if positive_event is not None:
        df['class'] = df['class'].apply(lambda x: positive_event in str(x)).astype(int)
    
    for col in df.columns:
        if col != 'class': 
            df[f"{col}__is_missing"] = 1 if df.isna().sum()[col] == df.shape[0] else 0

    # df['is_simulated'] = int('SIMULATED' in instance)
    # df['is_well'] = int('WELL' in instance)
    #df['is_drawn'] = int('DRAWN' in instance)
    
    df = df.fillna(0)

    if type(periods_undersample) in [int, str]:
        if type(periods_undersample) == int:
            periods_undersample = f'{periods_undersample}s'  
            
        df = df.resample(periods_undersample).last() 
        
    df = df.reset_index()
    return df
    

def preprocess_instances(df_metadata, periods_undersample, positive_event):
    addresses = df_metadata['Address'].tolist()
    ids = df_metadata['id'].tolist()
    instances = df_metadata['Instance'].tolist()
    df_list = []

    for address, id_, instance in zip(addresses, ids, instances):
        df = preprocess(pd.read_csv(address), instance, periods_undersample, positive_event)
        
        df['id'] = id_
        df_list.append(df)
    
    df = pd.concat(df_list, axis=0)
    df = df.merge(df_metadata, on='id', how='left').set_index('timestamp')
    
    return df