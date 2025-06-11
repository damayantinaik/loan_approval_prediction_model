"""
    It fetches a single record from the validation data and verifies the output using assert statements.

    It validates the following checks:
    The output is not null.
    The output data type is str.
    The output is Y for given data (fixed)
"""

# These checks are run using the pytest

# Import libraries

import pytest_

# Import files and modules
from prediction_model.config import config
from prediction_model.predict import make_prediction
from prediction_model.processing.data_management import load_dataset

@pytest.fixture
def single_prediction(): 
    """This function will make the prediction for a single record"""

    test_data = load_dataset(file_name = config.TEST_FILE)
    single_test = test_data[0:1]
    result = make_prediction(single_test)
    return result


# Test Prediction
def test_single_prediction_not_none(single_prediction):
    """This function will check if the result of the prediction is not None"""
    assert single_prediction is not None


def test_single_prediction_dtype(single_prediction):
    """This function will check if the data type of the result of the prediction is str i.e string"""
    assert isinstance(single_prediction.get('prediction')[0], str)


def test_single_prediction_output(single_prediction):
    '''This function will check if the result of the prediction is Y'''
    assert single_prediction.get('prediction')[0] =='Y' 