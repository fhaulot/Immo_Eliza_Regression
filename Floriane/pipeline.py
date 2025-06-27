import pandas as pd
import numpy as np

class Preprocessing : 

    def drop_columns(self, df, columns):
        """
        Drop specified columns from the DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame from which to drop columns.
        columns (list): List of column names to drop.
        
        Returns:
        pd.DataFrame: DataFrame with specified columns dropped.
        """
        return df.drop(columns=columns)