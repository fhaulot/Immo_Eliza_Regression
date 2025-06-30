import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Preprocessing : 

    def __init__(self):
        pass

    def read_csv(self, df, path):
        """
        Reading the cleaned_data csv file
        """
        df = pd.read_csv(path)
        print(df.info())
        return df
    
    def drop_columns(self, df):
        """
        Drop specified columns from the DataFrame.
        """
        Col_to_drop = ["url", 'locality', "MunicipalityCleanName"]
        cleaned_columns = df.drop(columns=Col_to_drop, axis=0)
        print(cleaned_columns.info())
        return cleaned_columns
    
    def ordinal_building_condition(self, df):
        """
        Perform ordinal encoding on the building condition using a mapping.
        """
        df = pd.DataFrame({"buildingCondition" : ["AS_NEW", "GOOD", "JUST_RENOVATED", "TO_RENOVATE", "TO_BE_DONE_UP", "TO_RESTORE"]})
        condition_order = {"AS_NEW": 1, "GOOD": 2, "JUST_RENOVATED": 3, "TO_RENOVATE": 4, "TO_BE_DONE_UP": 5, "TO_RESTORE": 6}
        df['encoded_condition'] = df['buildingCondition'].map(condition_order)
        return df
    
    def ordinal_epc_score(self, df) :
        """
        Perform ordinal encoding for the column of epc score using a mapping
        """
        df = pd.DataFrame({"epcScore" : ["A", "B", "C", "D", "E", "F", "G"]})
        epc_order = {"A" : 1, "B" : 2, "C" : 3, "D" : 4, "E" : 5, "F" : 6, "G" : 7}
        df['encoded_epc'] = df['epcScore'].map(epc_order)
    
    def categorical_encode(self, df):
        """
        Perform categorical encoding on specified columns.
        """
        encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
        for cat_par in (["type", "subtype", "province", "region"]):
            new_cols_df = encoder.fit_transform(df[cat_par])
            df = pd.concat([df, new_cols_df], axis=1).drop([cat_par], axis=1)
        return df
