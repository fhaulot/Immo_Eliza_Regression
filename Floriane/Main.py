from pipeline import Preprocessing

df = Preprocessing.read_csv('cleaned_data.csv')

df = Preprocessing.drop_columns()
df.info()

df = Preprocessing.ordinal_building_condition()
df.info()

df = Preprocessing.ordinal_epc_score()
df.info()

df = Preprocessing.categorical_encode()
df.info()