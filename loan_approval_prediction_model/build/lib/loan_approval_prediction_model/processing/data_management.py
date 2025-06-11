# This module contains functions required for loading the data, saving serialized ML model, 
# and loading deserialized ML model using joblib.

# import libraries

import os

import pandas as pd
import joblib

# Import other files and modules

from prediction_model.config import config

def load_dataset(file_name):
    """Read Data"""
    file_path = os.path.join(config.DATAPATH, file_name)
    _data = pd.read_csv(file_path)
    return _data


def save_pipeline(pipeline_to_save):
    """Store output of pipeline
       Exporting pickle file of trained Model"""
    
    save_file_name = 'classification_v1.pkl'

    save_path = os.path.join(config.SAVED_MODEL_PATH, save_file_name)
    joblib.dump(pipeline_to_save, save_path)
    print("Saved Pipeline : ", save_file_name)


def load_pipeline(pipeline_to_load):
    """Improving pickle file of trained Model"""
    save_path = os.path.join(config.SAVED_MODEL_PATH, pipeline_to_load)
    trained_model = joblib.load(save_path)
    return trained_model
        
