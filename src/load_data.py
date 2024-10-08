import os
import pandas as pd
import numpy as np

def process_metadata(dir_path, folder_classes=['3', '0'], file_format='.csv'):
    
    addresses, instances, events, y = [], [], [], []
    for root, _, files in os.walk(dir_path):
        
        # last name in the folder path is the class
        event = os.path.basename(root)
        if event in folder_classes:
            for file in files:
                if file.endswith(file_format):
                    address = os.path.join(root, file)
                    instance = file.split('.')[0]
                    
                    if len(folder_classes) == 2:
                        positive_class = folder_classes[0]
                        y_ = 1 if event == positive_class else 0
                    else:
                        y_ = event
                    addresses.append(address)
                    instances.append(instance)
                    events.append(event)
                    y.append(y_)
                    
    sorted_data = sorted(zip(addresses, instances, events, y), key=lambda item: item[0])
    addresses, instances, events, y = map(list, zip(*sorted_data))
    return addresses, instances, events, y