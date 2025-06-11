# Import Libraries
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMinin
from sklearn.preprocessing import LabelEncoder

# Import other files and/modules

from prediction_model.config import config

# The mean of the numerical variable is used here for imputing missing values.
# However, one can use other missing value imputation techniques such as stochastic regression imputation, 
# interpolation, and cold deck imputation

# Numerical Imputer

class NumericalImputer(BaseEstimator, TransformerMinin):
    """Numerical Data missing value imputer"""

    
    def __init__(self, variables = None):
        self.variables = variables

    
    def fit(self, X, y = None):
        self.imputer_dict_ = {}
        
        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mean()
        return self
    

    def transform(self, X):
        X = X.copy

        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace = True)
        return X
    
# The most frequent value of the feature is used for imputing the missing categorical
# values, but you can also use other missing value imputtaion techniques, such as 
# missing indicator imputation, Multivariate Imputation by Chained Equation(MICE) 
# algorithm, and systematic random sampling imputation.

class CategoricalImputer(BaseEstimator, TransformerMinin):
    """Categorical Data Missing Value Imputer"""

    def __init__(self, variables= None):
        self.variables = variables

 
    def fit(self, X, y, None):
        self.imputer_dict_={}

        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mode()[0]
            return self

    def transform(self, X):
        X = X.copy

        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature], inplace = True)
            return x        
        
# Categorical Encoder
class CatgoricalEncoder(BaseEstimator, TransformerMinin):
    """Categorical Data Encoder"""

    def __init__(self, variables = None):
        self.variables = variables

    
    def fit(self, X, y):
        self.encoder_dict_ = {}

        for var in self.variables:
            t = X[var].value_counts().sort_values(ascending = True)

            self.encoder_dict_[var] = {k:i for i, k in enumerate(t, 0)}
        return self
    
    
    def transform(self, X):
        X = X.copy()

        # This partassumes that the encoder does not introduce  NANs
        # In that case, a check needs to be done and the code should break

        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])
        return X 
    
#  Create some new features

# Temporal features

class TemporalVariableEstimator(BaseEstimator, TransformerMixin):
    """Feature Engineering"""

    def __init__(self, variables = None, reference_variable = None):
        self.variables = variables
        self.reference_variable = reference_variable


    def fit(self, X, y = None):
        # No need to put anything, needed for sklearn Pipeline
        return self
    
    def transform(self, X):
        X = X.copy()

        for var in self.variables:
            X[var] = X[var ]+ X[self.reference_variable]
       
        return X
    
# Apply log trasnformation on variables for making patterns in the data more interpretable

class LogTransformation(BaseEstimator, TransformerMixin):
    """Transforming variables using Log Transformations"""

    def __init__(self, variables=None):
        self.variables = variables

     
    def fit(self, X, y):
        return self
    
    # Need to check in advance if the feature are <=0
    # If yes, needs to be transformed properly (e.g, np.log1p(X[var]))

    def transform(self, X):
        X = X.copy
        
        for var in self.variables:
            X[var] = np.log(X[var])
    return X

# Drop the features are insignificant to the model

class DropFeatures(BaseEstimator, TransformerMixin):
    """Dropping Features whic are less significant"""

    def __init__(self, variables_to_drop = None):
        self.variables_to_drop = variables_to_drop

    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X = X.drop(self.variables_to_drop, axis = 1)

        return X






