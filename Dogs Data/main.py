from dataviz import DataAnalysis 

def main():
    dataviz = DataAnalysis()
    df = dataviz.read_csv("./immoweb-dataset.csv")
    
    df = dataviz.convert_has_garden(df)
    
    df = dataviz.convert_garden_surface (df)
    
    df = dataviz.convert_has_terrace (df)
    
    df = dataviz.add_parking_col(df)
    
    df = dataviz.drop_column(df)
        
    df = dataviz.remove_empty_rows(df)

    df= dataviz.normalize_municipality(df)
            
    df = dataviz.sanitize_epcScore(df)
    

    df = dataviz.add_region_column(df)

    df = dataviz.price_square_meter(df)

    dataviz.save_csv(df, 'cleaned_data_with_outliers.csv')

    df = dataviz.filter_likely_outliers(df)

    dataviz.save_csv(df, 'cleaned_data.csv')
    
    print(df.info())
        

if __name__ == "__main__":
    main()

