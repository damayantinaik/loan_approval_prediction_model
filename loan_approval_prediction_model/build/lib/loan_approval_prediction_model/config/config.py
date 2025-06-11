# Import libraries

import pathlib
import os
import prediction_model

PACKAGE_ROOT = pathlib.path(prediction_model.__file__).resolve().parent

DATAPATH = os.path.join(PACKAGE_ROOT, 'datasets')
SAVED_MODEL_PATH = os.path.join(PACKAGE_ROOT, 'trained_models')

TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

TARGET = 'Loan_Status'

# features to keep
# Final features to keep in the data
FEATURES = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome',
            'Loan_Amount_Term', 'Credit_History', 'Property_Area']

NUMERICAL_FEATURES = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']

CATEGORICAL_FEARURES = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',   'Credit_History', 'Property_Area']

FEATURES_TO_ENCODE = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
TEMPORAL_FEATURES = ['ApplicantIncome']
TEMPORAL_ADDITION = 'ApplicantIncome'
LOG_FEATURES = ['ApplicantIncome', 'LoanAmount'] # Feature for log transform

DROP_FEATURES = ['CoApplicantIncome']
