import pandas as pd
import numpy as np


class DataAnalysis:
   
    def read_csv(self, path):
        # read the csv file with pandas
        df = pd.read_csv(path)
        print(df.info())
        return df

    def drop_column(self, df) :
        # dropping the columns we think don't have enough relevant data to use
        Col_to_drop = ["Unnamed: 0",
        "bathroomCount",
        "roomCount",
        "monthlyCost",
        "hasAttic",
        "hasBasement",
        "hasDressingRoom",
        "diningRoomSurface",
        "hasDiningRoom",
        "buildingConstructionYear",
        "facedeCount",
        "floorCount",
        "streetFacadeWidth",
        "hasLift",
        "floodZoneType",
        "heatingType",
        "hasHeatPump",
        "hasPhotovoltaicPanels",
        "hasThermicPanels",
        "kitchenSurface",
        "kitchenType",
        "landSurface",
        "hasLivingRoom",
        "livingRoomSurface",
        "hasBalcony",
        "gardenOrientation",
        "hasAirConditioning",
        "hasArmoredDoor",
        "hasVisiophone",
        "hasOffice",
        "toiletCount",
        "hasSwimmingPool",
        "hasFireplace",
        "terraceSurface",
        "terraceOrientation",
        "accessibleDisabledPeople", "parkingCountIndoor", "parkingCountOutdoor"]
        cleaned_columns = df.drop(columns=Col_to_drop, axis=1)
        print(cleaned_columns.info())
        return cleaned_columns


    # check "hasGarden" column, if "TRUE" change to 1, otherwise make it 0
    def convert_has_garden (self, df):         
        df["hasGarden"]= np.where(df["hasGarden"] == True,1,0)
        return df

    # check "gardenSurface" column, if empty change to 0, leave current value if NOT empty
    def convert_garden_surface (self, df):
        df["gardenSurface"]= df["gardenSurface"].fillna(0)
        return df

    # check "hasTerrace" column, if "TRUE" change to 1 otherwise make it 0
    def convert_has_terrace (self, df):         
        df["hasTerrace"]= np.where(df["hasTerrace"] == True,1,0)
        return df


    def save_csv(self, df, path):
        df.to_csv(path, index=False)

    def add_parking_col(self, df):
        df = self.__clean_parking_data(df)
        df["hasParking"] = df.apply(
            lambda row: 1 if (
                (not pd.isna(row.parkingCountIndoor) and row.parkingCountIndoor > 0)
                or (not pd.isna(row.parkingCountOutdoor) and row.parkingCountOutdoor > 0)
            ) else 0,
            axis=1
        )
        count_0 = df['hasParking'].value_counts().get(0)
        count_1 = df['hasParking'].value_counts().get(1)
        print('count_0', count_0, 'count_1', count_1 )
        print(df.info())
        return df

    def __check_if_too_large(self, df, col_name):
        return df[col_name] > 1000

    def __check_if_large_and_apt(self, df, col_name):
        return (df[col_name] > 100) & (df['type'] == 'APARTMENT')

    def __clean_parking_data(self, df):
        orig = df.shape[0]
        # > 1000 drop, >100 keep if type is not apartment
        rows_to_drop = (self.__check_if_too_large(df, 'parkingCountIndoor')) | (self.__check_if_too_large(df, 'parkingCountOutdoor'))
        df = df[~rows_to_drop]
        count = orig - df.shape[0]
        print(f'Dropped {count} too large')
        
        orig = df.shape[0]
        rows_to_drop = (self.__check_if_large_and_apt(df, 'parkingCountIndoor')) | (self.__check_if_large_and_apt(df, 'parkingCountOutdoor'))
        df = df[~rows_to_drop]
        count = orig - df.shape[0]
        print(f'Dropped {count} large and apartment type ')
        return df
    
    def remove_empty_rows(self, df):
        cleaned_df = df.dropna()
        return cleaned_df

    def filter_likely_outliers(self, df):
        return df[
            (df['price'] >= 50000) & (df['price'] <= 1_000_000) &
            (df['habitableSurface'] >= 15) & (df['habitableSurface'] <= 600) &
            (df['bedroomCount'] <= 20) 
            & ((df['gardenSurface'] <= 3000) | (df['hasGarden'] == 0))
        ]

    # Normalize Municipality Name
    def normalize_municipality (self, df):
        postcode_df = pd.read_excel("./PC_Reference.xlsx")                                                        
        normalized_df = pd.merge(df, postcode_df, on='postCode', how='left')
        return normalized_df

    def sanitize_epcScore(self, df):
        # Matches scores from A to G with optional + and -
        valid_epc_regex = r'^[A-G][+-]*$'
        df = df[df['epcScore'].str.match(valid_epc_regex)]
        return df

    def add_region_column(self, df) :
        province_to_region = {'Antwerp': 'Flanders', 
                              'East Flanders': 'Flanders',
                              'Flemish Brabant': 'Flanders', 
                              'Limburg': 'Flanders',
                              'West Flanders': 'Flanders',
                              'Walloon Brabant': 'Wallonia',
                              'Hainaut': 'Wallonia',
                              'Liège': 'Wallonia',
                              'Luxembourg': 'Wallonia',
                              'Namur': 'Wallonia',
                              'Brussels': 'Brussels'}
        df['region'] = df['province'].map(province_to_region)
        return df


    # Add price per square meter
    def price_square_meter(self, df):
        df["price_square_meter"] = df["price"]/df["habitableSurface"]
        print(df.info())
        return df

