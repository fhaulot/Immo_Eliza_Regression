from pipeline import Preprocessing
import pandas as pd
df = pd.read_csv('./Dogs Data/Cleaned_data.csv')

preprocessing = Preprocessing()
df = preprocessing.drop_columns(df)
df.info()

df = preprocessing.ordinal_building_condition(df)
df.info()

df = preprocessing.ordinal_epc_score(df)
df = preprocessing.categorical_encode(df)
df.info()