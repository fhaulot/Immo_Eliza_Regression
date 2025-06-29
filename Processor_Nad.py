import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Processor_Nad:
    """
    Removes extreme outliers and replaces categorial parameters.
    It assumes that no na (empty cells) values are present.
    """

    def __init__(self, name_raw_csv, name_postcodes_geo, name_processed_csv):
        # Name + path from the current directory is expected
        self.name_raw_csv=name_raw_csv
        self.name_processed_csv=name_processed_csv
        self.name_postcodes_geo=name_postcodes_geo

    def process(self):

        ############ Process Floriane's semi-raw file (na values filled)

        # Read raw data
        df = pd.read_csv(self.name_raw_csv)

        # Drop url, locality columns
        df = df.drop(columns=["id","url","locality","MunicipalityCleanName"], axis=1)

        #print(df['postCode'].nunique())

        # Replace postal code with latitude, longitude from external csv file - maybe need to leave only one row per postal code
        # in the postal code file before merging ??? And for cases of two postal codes per row - leave only one, but make sure the second one
        # is also given in a separate row, if with same coords - ok; in the end check one postal code per row should be present,
        # allowing  same coord-s per different postcal codes ?????
        df_postcode=pd.read_csv(self.name_postcodes_geo)
        df_postcode.astype({'postCode': 'int64'})
        #df_postcode.info()
        df = pd.merge(df, df_postcode[['postCode','lat', 'lng']], on='postCode', how='left')
        df = df.drop(columns=["postCode"], axis=1)

        print(df['epcScore'].unique())

        # Map building condition categories to numerical values
        cond_map = {
        'TO_RESTORE': 1,
        'TO_RENOVATE': 2, 
        'TO_BE_DONE_UP': 3, 
        'GOOD': 4, 
        'JUST_RENOVATED': 5, 
        'AS_NEW': 6,
        }
        df['bldCondition'] = df['buildingCondition'].map(cond_map)
        #print(df[(df['condition']==2) & (df['buildingCondition'] == 'TO_RESTORE') ])
        df = df.drop(columns=["buildingCondition"], axis=1)

        # Map epcScore categories to numerical values
        score_map = {
        'G': 1,
        'F': 2, 
        'E': 3, 
        'D': 4, 
        'C': 5, 
        'B': 6,
        'A': 7,
        'A+': 8,
        'A++': 9
        }
        df['eScore'] = df['epcScore'].map(score_map)
        df = df.drop(columns=["epcScore"], axis=1)

        df.info()
    
        # Encode categorical features: type, subtype, province, region - maybe need to drop province, region ?????
        encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
        for cat_par in (["type", "subtype", "province", "region"]):
            new_cols_df = encoder.fit_transform(df[[cat_par]])
            df = pd.concat([df, new_cols_df], axis=1).drop([cat_par], axis=1)

        # Save to a new csv file
        #df_processed=df.to_csv(self.name_processed_csv, encoding='utf-8', index=False)
        #df.info()
    
        # Split into trainig and testing: Jean  - do we have to predict price or price per m2 ????????????
        X = df.drop('price', axis=1) 
        y = df['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=41, test_size=0.2) # 20% of rows will go to the testing sample, 80% to the training
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Titanic tutorial why 41 or 42 ???????
        print("Shape X_train, X_test, y_train, y_shape: ", type(X_train), X_test.shape, type(y_train), y_test.shape)

        # Get mean and standard deviation from training set (per feature) and normalize training and testing sets using these values for the training set
        for idx, name in enumerate(list(X_train.columns.values)):
            print(f"Training set: '{name}' has mean {X_train[name].mean():.2f} and standard deviation {X_train[name].std():.2f}")
            print(f"Testing set: '{name}' has mean {X_test[name].mean():.2f} and standard deviation {X_test[name].std():.2f}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        X_train_norm = (X_train - X_train.mean()) / X_train.std()  
        X_test_norm = (X_test - X_train.mean()) / X_train.std()
        # --- To check that normalisation of training set was good, should get mean=0 , std=1 for all columns of the training set below
        # (and nearly these values for the testing set)
        for idx, name in enumerate(list(X_train.columns.values)):
            print(f"Training set '{name}' has mean {X_train_norm[name].mean():.2f} and standard deviation {X_train_norm[name].std():.2f}") 
            print(f"Testing set: '{name}' has mean {X_test_norm[name].mean():.2f} and standard deviation {X_test_norm[name].std():.2f}")

        #scaler = StandardScaler()
        #data[['standardized_column']] = scaler.fit_transform(data[['numeric_column']])

        #from sklearn.preprocessing import StandardScaler
        #scaler = StandardScaler()
        #X_scaled = scaler.fit_transform(X)


        #df.info()
            
        

if __name__ == "__main__":
    processor=Processor_Nad("./Dogs Data/Cleaned_data.csv", "./Data/belgian-cities-geocoded.csv", "./Data/Processed_data.csv")
    processor.process()
            
