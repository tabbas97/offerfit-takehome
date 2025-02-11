from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline

class DataModeler:
    
    def __init__(self, sample_df: pd.DataFrame):
        '''
        Initialize the DataModeler as necessary.
        '''
        # ** Your code here **
        self.source_df : pd.DataFrame = sample_df.copy(deep=True) # Preserve the original data model
        self.train_df : pd.DataFrame = None

    def prepare_data(self, oos_df: pd.DataFrame = None, allow_implicit_data_conversions = True) -> pd.DataFrame:
        '''
        Prepare a dataframe so it contains only the columns to model and having suitable types.
        If the argument is None, work on the training data passed in the constructor.
        '''
        # ** Your code here **
        if oos_df is None:
            # Assuming all the columns in the sample data are to be used for modeling
            # Excluding the columns with object data type
            self.train_df = self.source_df.copy(deep=True)
            
            # Extract labels from the training data
            self.train_labels = self.train_df['outcome'].copy()
            
            # Drop the outcome column from the training data 
            self.train_df.drop(columns=['outcome'], inplace=True)
            
            # Drop the customer_id since that is not a feature
            self.train_df.drop(columns=['customer_id'], inplace=True)
            
            # Convert the transaction_date to a numeric value
            self.train_df['transaction_date'] = self.train_df.transaction_date.apply(
                lambda x: datetime.strptime(x, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()*10**9 if x is not None else None
                )
            
        else:
            
            oos_copy = oos_df.copy(deep=True) # Avoid any data leakage back to the source dataframe
            if allow_implicit_data_conversions:
                # Allow conversion only for amount column -> can come from any lower precision to float64
                if 'amount' in oos_copy and pd.api.types.is_numeric_dtype(oos_copy['amount']):
                    oos_copy['amount'] = oos_copy['amount'].astype(np.float64)
            
            # Validate the data types of the columns
            # Check if the columns are in the dataframe and are of the right data type
            for req_col in ['amount', 'transaction_date']:
                
                if req_col not in oos_copy:
                    raise ValueError(f"Column '{req_col}' is missing in the dataframe")
                
                if oos_copy[req_col].dtype != self.source_df[req_col].dtype:
                    raise ValueError(f"Column '{req_col}' is not of the right data type")
            
            # Convert the transaction_date to a numeric value
            oos_copy['transaction_date'] = oos_copy.transaction_date.apply(
                lambda x: datetime.strptime(x, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()*10**9 if x is not None else None
                )
            
            # Drop the customer_id since that is not a feature
            if 'customer_id' in oos_copy:
                oos_copy.drop(columns=['customer_id'], inplace=True)
                
            return oos_copy

    def impute_missing(self, oos_df: pd.DataFrame = None) -> pd.DataFrame:
        '''
        Fill any missing values with the appropriate mean (average) value.
        If the argument is None, work on the training data passed in the constructor.
        Hint: Watch out for data leakage in your solution.
        '''
        # ** Your code here **
        if oos_df is None:
            # Impute missing values in the training data
            self.train_df.fillna(self.train_df.mean(), inplace=True)
        else:
            # Impute missing values in the out-of-sample data
            oos_copy = oos_df.copy(deep=True)
            oos_copy.fillna(oos_copy.mean(), inplace=True)
            return oos_copy

    def fit(self) -> None:
        '''
        Fit the model of your choice on the training data paased in the constructor, assuming it has
        been prepared by the functions prepare_data and impute_missing
        '''
        # ** Your code here **
        # Fit the model on the training data
        self.model = make_pipeline(
            # StandardScaler(),             # A scaling step is not necessary for the AdaBoostClassifier nor the GaussianNB 
                                            # since they are not sensitive to the scale of the features like KNN or SVM
            # GaussianNB(),                 # -> Only achieved 70% accuracy
            AdaBoostClassifier(n_estimators=10, random_state=0)
            )
        # self.model = GaussianNB()
        self.model.fit(self.train_df, self.train_labels)

    def model_summary(self) -> str:
        '''
        Create a short summary of the model you have fit.
        '''
        # ** Your code here **
        return str({
            'model_type': 'AdaBoostClassifier',
            'model' : self.model
        })

    def predict(self, oos_df: pd.DataFrame = None) -> pd.Series[bool]:
        '''
        Make a set of predictions with your model. Assume the data has been prepared by the
        functions prepare_data and impute_missing.
        If the argument is None, work on the training data passed in the constructor.
        '''
        # ** Your code here **
        if oos_df is None:
            # Predict on the training data
            return self.model.predict(self.train_df)
        else:
            # Predict on the out-of-sample data
            return self.model.predict(oos_df)

    def save(self, path: str) -> None:
        '''
        Save the DataModeler so it can be re-used.
        '''
        # ** Your code here **
        joblib.dump(self, path) # The recommended way of saving scikit-learn models

    @staticmethod
    def load(path: str) -> DataModeler:
        '''
        Reload the DataModeler from the saved state so it can be re-used.
        '''
        # ** Your code here **
        return joblib.load(path)


#################################################################################
# You should not have to modify the code below this point

transact_train_sample = pd.DataFrame(
    {
        "customer_id": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "amount": [1, 3, 12, 6, 0.5, 0.2, np.nan, 5, np.nan, 3],
        "transaction_date": [
            '2022-01-01',
            '2022-08-01',
            None,
            '2022-12-01',
            '2022-02-01',
            None,
            '2022-02-01',
            '2022-01-01',
            '2022-11-01',
            '2022-01-01'
        ],
        "outcome" : [False, True, True, True, False, False, True, True, True, False]
    }
)


print(f"Training sample:\n{transact_train_sample}\n")

# <Expected Output>
# Training sample:
#    customer_id  amount transaction_date  outcome
# 0           11     1.0       2022-01-01    False
# 1           12     3.0       2022-08-01     True
# 2           13    12.0             None     True
# 3           14     6.0       2022-12-01     True
# 4           15     0.5       2022-02-01    False
# 5           16     0.2             None    False
# 6           17     NaN       2022-02-01     True
# 7           18     5.0       2022-01-01     True
# 8           19     NaN       2022-11-01     True
# 9           20     3.0       2022-01-01    False


print(f"Current dtypes:\n{transact_train_sample.dtypes}\n")

# <Expected Output>
# Current dtypes:
# customer_id           int64
# amount              float64
# transaction_date     object
# outcome                bool
# dtype: object

transactions_modeler = DataModeler(transact_train_sample)

transactions_modeler.prepare_data()

print(f"Changed columns to:\n{transactions_modeler.train_df.dtypes}\n")

# <Expected Output>
# Changed columns to:
# amount              float64
# transaction_date    float64
# dtype: object

transactions_modeler.impute_missing()

print(f"Imputed missing as mean:\n{transactions_modeler.train_df}\n")

# <Expected Output>
# Imputed missing as mean:
#               amount  transaction_date
# customer_id
# 11            1.0000      1.640995e+18
# 12            3.0000      1.659312e+18
# 13           12.0000      1.650845e+18
# 14            6.0000      1.669853e+18
# 15            0.5000      1.643674e+18
# 16            0.2000      1.650845e+18
# 17            3.8375      1.643674e+18
# 18            5.0000      1.640995e+18
# 19            3.8375      1.667261e+18
# 20            3.0000      1.640995e+18


print("Fitting  model")
transactions_modeler.fit()

print(f"Fit model:\n{transactions_modeler.model_summary()}\n")

# <Expected Output>
# Fitting  model
# Fit model:
# <<< ANY SHORT SUMMARY OF THE MODEL YOU CHOSE >>>

in_sample_predictions = transactions_modeler.predict()
print(f"Predicted on training sample: {in_sample_predictions}\n")
print(f'Accuracy = {sum(in_sample_predictions ==  [False, True, True, True, False, False, True, True, True, False])/.1}%')

# <Expected Output>
# Predicting on training sample [False  True  True  True False False True  True  True False]
# Accuracy = 100.0%

transactions_modeler.save("transact_modeler")
loaded_modeler = DataModeler.load("transact_modeler")

print(f"Loaded DataModeler sample df:\n{loaded_modeler.model_summary()}\n")

# <Expected Output>
# Loaded DataModeler sample df:
# <<< THE SUMMARY OF THE MODEL YOU CHOSE >>>

transact_test_sample = pd.DataFrame(
    {
        "customer_id": [21, 22, 23, 24, 25],
        "amount": [0.5, np.nan, 8, 3, 2],
        "transaction_date": [
            '2022-02-01',
            '2022-11-01',
            '2022-06-01',
            None,
            '2022-02-01'
        ]
    }
)

adjusted_test_sample = transactions_modeler.prepare_data(transact_test_sample)

print(f"Changed columns to:\n{adjusted_test_sample.dtypes}\n")

# <Expected Output>
# Changed columns to:
# amount              float64
# transaction_date    float64
# dtype: object

filled_test_sample = transactions_modeler.impute_missing(adjusted_test_sample)

print(f"Imputed missing as mean:\n{filled_test_sample}\n")

# <Expected Output>
# Imputed missing as mean:
#              amount  transaction_date
# customer_id
# 21           0.5000      1.643674e+18
# 22           3.8375      1.667261e+18
# 23           8.0000      1.654042e+18
# 24           3.0000      1.650845e+18
# 25           2.0000      1.643674e+18

oos_predictions = transactions_modeler.predict(filled_test_sample)
print(f"Predicted on out of sample data: {oos_predictions}\n")
print(f'Accuracy = {sum(oos_predictions == [False, True, True, False, False])/.05}%')

# <Expected Output>
# Predicted on out of sample data: [False True True False False] ([0 1 1 0 0])
# Accuracy = 100.0%