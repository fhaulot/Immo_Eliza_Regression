from pipeline import Preprocessing
import pandas as pd

# This script is used to preprocess the data before running the regression models.
preprocessing = Preprocessing()

df = preprocessing.read_csv(None, r"C:\Users\fhaul\Documents\GitHub\Immo_Eliza_Regression\Floriane\cleaned_data.csv")  

df = preprocessing.drop_columns(df)

df = preprocessing.ordinal_building_condition(df)

df = preprocessing.ordinal_epc_score(df)

df = preprocessing.categorical_encode(df)

df = preprocessing.postcode_geo(df)

df.head(5)

# Save the preprocessed DataFrame to a new CSV file
df.to_csv(r"C:\Users\fhaul\Documents\GitHub\Immo_Eliza_Regression\Floriane\preprocessed_data.csv", index=False)