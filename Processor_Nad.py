import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



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

        self.X_train_norm=None
        self.X_test_norm=None
        self.y_train=None
        self.y_test=None

    def process(self):

        ############ Prepare Floriane's semi-raw file (na values filled) for regression model

        # Read raw data
        df = pd.read_csv(self.name_raw_csv)

        print(len(df))

        # Drop url, locality columns, and price per m2 !!!!!!!!!!!!!!!!!!!!
        df = df.drop(columns=["id","url","locality","MunicipalityCleanName", "price_square_meter"], axis=1)
        #df = df.drop(columns=["id","url","locality","MunicipalityCleanName", "price_square_meter","hasGarden","type","region", "province"], axis=1)
 
        #print(df['postCode'].nunique())

        # Replace postal code with latitude, longitude from external csv file 
        df_postcode=pd.read_csv(self.name_postcodes_geo)
        df_postcode = df_postcode.drop_duplicates(subset='postCode', keep='first') # remove duplicate postal codes, only keep first row (first set of coords)
        df_postcode.astype({'postCode': 'int64'})
        #df_postcode.info()
        df = pd.merge(df, df_postcode[['postCode','lat', 'lng']], on='postCode', how='left')
        df = df.drop(columns=["postCode"], axis=1)

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

        print(df.info())

        # Encode categorical features: type, subtype, province, region - maybe need to drop province, region ?????
        encoder = OneHotEncoder(sparse_output=False).set_output(transform="pandas")
        for cat_par in (["type", "subtype", "province", "region"]): # !!!!!!!!!!!!!!!!
        #for cat_par in (["subtype","province"]): # !!!!!!!!!!!!!!!!
        #for cat_par in (["subtype"]):    
            new_cols_df = encoder.fit_transform(df[[cat_par]])
            df = pd.concat([df, new_cols_df], axis=1).drop([cat_par], axis=1)

        # Save to a new csv file
        #df_processed=df.to_csv(self.name_processed_csv, encoding='utf-8', index=False)
        #df.info()
        
        # Split into trainig and testing: Jean  - do we have to predict price or price per m2 ????????????
        X = df.drop('price', axis=1) 
        y = df['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=41, test_size=0.2) # 20% of rows will go to the testing sample, 80% to the training
        print("Shape X_train, X_test, y_train, y_shape: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        # Get mean and standard deviation from training set (per feature) and normalize training and testing sets using these values for the training set
        #for idx, name in enumerate(list(X_train.columns.values)):
        #    print(f"Training set: '{name}' has mean {X_train[name].mean():.2f} and standard deviation {X_train[name].std():.2f}")
        #    print(f"Testing set: '{name}' has mean {X_test[name].mean():.2f} and standard deviation {X_test[name].std():.2f}")
        #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        X_train_norm = (X_train - X_train.mean()) / X_train.std()  
        X_test_norm = (X_test - X_train.mean()) / X_train.std()
        # --- To check that normalisation of training set was good, should get mean=0 , std=1 for all columns of the training set below
        # (and nearly these values for the testing set)
        #for idx, name in enumerate(list(X_train.columns.values)):
        #    print(f"Training set '{name}' has mean {X_train_norm[name].mean():.2f} and standard deviation {X_train_norm[name].std():.2f}") 
        #    print(f"Testing set: '{name}' has mean {X_test_norm[name].mean():.2f} and standard deviation {X_test_norm[name].std():.2f}")

        self.X_train_norm=X_train_norm
        self.X_test_norm=X_test_norm
        self.y_train=y_train
        self.y_test=y_test


    def fitLinearRegr(self):

        ############ Fit linear regression


        X_train=self.X_train_norm
        X_test=self.X_test_norm
        y_train=self.y_train
        y_test=self.y_test

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # print model coeff-s
        #for i in range(0, len(model.coef_)):
        #    print(model.feature_names_in_[i], model.coef_[i])
        #print("Model intercept:", model.intercept_)

        # Use the model on the testing set

        y_test_pred = model.predict(X_test)
        prices_test_pred_df= pd.DataFrame({'actual price (EUR)': y_test, 'predicted price (EUR)': y_test_pred})
        #print(prices_test_pred_df)
 
        mae_test = mean_absolute_error(y_test, y_test_pred)
        r2_test = r2_score(y_test, y_test_pred)

        print("Mean Absolute Error (MAE) test set:", mae_test)
        print("R-squared Score test set:", r2_test)

        # Use the model on the training set
        
        y_train_pred = model.predict(X_train)
        prices_train_pred_df= pd.DataFrame({'actual price (EUR)': y_train, 'predicted price (EUR)': y_train_pred})
        #print(prices_train_pred_df)
 
        mae_train = mean_absolute_error(y_train, y_train_pred)
        r2_train = r2_score(y_train, y_train_pred)

        print("Mean Absolute Error (MAE) train set:", mae_train)
        print("R-squared Score train set:", r2_train)

        # Make scatter plot to see goodness of the fit

        """ ax = prices_test_pred_df.plot.scatter( x='actual price for test set', y='predicted price for test set',
                                                marker='o',      # Marker type, e.g., 'o', 'x', '^', etc.
                                                color='blue',     # Marker color, e.g., 'red', 'blue', '#123456'
                                                s=10           # Marker size (optional)
                                )
        ax=prices_train_pred_df.plot.scatter( x='actual price for training set', y='predicted price for training set',
                                                marker='o',      # Marker type, e.g., 'o', 'x', '^', etc.
                                                color='yellow',     # Marker color, e.g., 'red', 'blue', '#123456'
                                                s=10            # Marker size (optional)
                                ) """
        
        df1 = prices_train_pred_df.copy()
        df2 = prices_test_pred_df.copy()
        df1['Dataset:'] = 'training'
        df2['Dataset:'] = 'test'        
        df_combined = pd.concat([df1, df2])
        custom_markers = {'training': '+', 'test': 'x'}         # Circle for A, square for B
        custom_palette = {'training': 'blue', 'test': 'red'}    # Red for A, blue for B
        ax= sns.scatterplot(
            data=df_combined,
            x='actual price (EUR)', 
            y='predicted price (EUR)',
            style='Dataset:',         # Different marker for each DataFrame
            hue='Dataset:',            # Different color for each DataFrame
            palette=custom_palette,
            markers=custom_markers,
            s=15
        )
        xmin, xmax = plt.xlim()
        # For slope 1 and intercept 0 (y = x)
        plt.plot([xmin, xmax], [xmin, xmax], color='black', linestyle='-',  lw=1, label='predicted = actual')
        plt.legend()
        ax.set_title(f'Model: Linear regression,\n N_params: 13[14], N_rows: 41726, N_test_frac: 0.2,\n R2_test: {r2_test:.3f}, R2_train: {r2_train:.3f},\n MAE_test: {round(mae_test)}, MAE_train: {round(mae_train)}', 
                fontsize=7,           # Font size
                color='green',       # Font color
                #fontname='Arial',      # Font family
                fontstyle='italic')    # Font style (optional, e.g., 'italic')
                #fontweight='bold'  )

        plt.show()



        
        

if __name__ == "__main__":
    processor=Processor_Nad("./Dogs Data/Cleaned_data.csv", "./Data/belgian-cities-geocoded.csv", "./Data/Processed_data.csv")
    processor.process()
    processor.fitLinearRegr()
            
