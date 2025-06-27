import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd

# File with property data:
df = pd.read_csv("./Data/Cleaned_data.csv")

# File linking postal codes with municipality names, provided by Emmanuel:
# I had to slightly modify it: replace ; with , as separator, and change column name from Post code to postcode for join function to work.
pc = pd.read_csv("./Data/cities2_formatted.csv")

############ Dropping extreme values following Emmanuel's study of outliers

df = df[
    ((df['type'] == 'apartment') & (df['habitablesurface'] < 5000)) |
    ((df['type'] == 'house') & (df['habitablesurface'] < 10100))]

df = df[
    ((df['subtype'] == 'house') & (df['price_per_m2'] < 20000)) |
    (df['subtype'] != 'house')]

df = df[
    ((df['subtype'] == 'villa') & (df['price_per_m2'] < 20000)) |
    (df['subtype'] != 'villa')]

#print(df[df["postcode"].isin([3000])]["price"]) # print localities with a given postal code 

############ Finding most and least-expensive properties per m2 in Flanders and Wallonia,
############ using median-averaged values per postal code

# Post codes of municipalities in Flanders and Wallonia separately
postcode_range_fl=np.arange(1500,4000),np.arange(8000,10000)
postcode_range_wl=np.arange(1300,1500),np.arange(4000,8000)

# Merge the propery data frame with the postal code frame based on the postal code, work further with this merged data frame (dfj)
#Examples of municipalities with multiple postalcodes: Heverlee: 3000,3001;  Ramskapelle: 8301,8620; Saint-Gillis: 1050,1060; Antwerpen: 2000,2018,2020,2030,...
#Examples of postalcodes with several municipalities: 3000: Leuven,Heverlee; 1050: Saint-Gillis,Ixelles 
dfj=pd.merge(df, pc, how='left', on="postcode")

# Utility function to gather all municipalities with the same postal code into one string shown in output
def custom_aggregation_funcion(vals): return str(set(vals)).strip('{}').replace("'","")

####### --------- Print on screen

# Uncomment header line depending on the region (Flanders/Wallonia/Belgium)
print("*********************************************************************************\n"+ 
#"10 MOST EXPENSIVE municipalities in Flanders based on ~~~ median price per m2 ~~~: \n"+
"10 MOST EXPENSIVE municipalities in Wallonia based on ~~~ median price per m2 ~~~: \n"+
#"10 MOST EXPENSIVE municipalities in Belgium based on ~~~ median price per m2 ~~~: \n"+
"*********************************************************************************")
print(
    dfj[dfj["postcode"].isin(np.concatenate(postcode_range_wl))].groupby('postcode').agg( # postcode_range_wl for Wallonia, postcode_range_fl for Flanders
    #dfj.groupby('postcode').agg( # use this line instead of the line above for the whole of Belgium        
    municipality=("Municipality",custom_aggregation_funcion), # new column!  
    median_price_pm2=("price_per_m2", "median"),
    mean_price_pm2=("price_per_m2", "mean"),
    median_price=("price", "median"), 
    mean_price=("price", "mean")
).nlargest(10,"median_price_pm2") 
)

print()

# Uncomment header line depending on the region (Flanders/Wallonia/Belgium)
print("*********************************************************************************\n"+ 
#"10 LEAST EXPENSIVE municipalities in Flanders based on ~~~ median price per m2 ~~~: \n"+
"10 LEAST EXPENSIVE municipalities in Wallonia based on ~~~ median price per m2 ~~~: \n"+
#"10 LEAST EXPENSIVE municipalities in Belgium based on ~~~ median price per m2 ~~~: \n"+
"*********************************************************************************")
print(
    dfj[dfj["postcode"].isin(np.concatenate(postcode_range_wl))].groupby('postcode').agg( # postcode_range_wl for Wallonia, postcode_range_fl for Flanders
    #dfj.groupby('postcode').agg( # use this line instead of the line above for the whole of Belgium        
    municipality=("Municipality",custom_aggregation_funcion), # new column!  
    median_price_pm2=("price_per_m2", "median"),
    mean_price_pm2=("price_per_m2", "mean"),
    median_price=("price", "median"), 
    mean_price=("price", "mean")
).nsmallest(10,"median_price_pm2")
)

print()

####### --------- Compare 10 most or least expensive municipalities in Flanders and Wallonia in the horizontal bar plot

ax=sns.barplot(data=dfj[dfj["postcode"].isin(np.concatenate(postcode_range_fl))].groupby('postcode').agg(
    municipality=("Municipality",custom_aggregation_funcion), # new column!  
    median_price_pm2=("price_per_m2", "median")
).nsmallest(10,"median_price_pm2"), x="median_price_pm2", y="municipality") # choose between .nsmallest and .nlargest; change 10 to ~2000 for all municipalities
sns.barplot(data=dfj[dfj["postcode"].isin(np.concatenate(postcode_range_wl))].groupby('postcode').agg(
    municipality=("Municipality",custom_aggregation_funcion), # new column!  
    median_price_pm2=("price_per_m2", "median")
).nsmallest(10,"median_price_pm2"), x="median_price_pm2", y="municipality", ax=ax) # choose between .nsmallest and .nlargest; change 10 to ~2000 for all municipalitie
#plt.gca().set_yticklabels([]) # uncomment in case you want to plot all municipalities and don't want the overlapping labels on the y axis
#plt.gca().invert_yaxis() # uncomment if you want to change order of bars - shortest at the top
plt.gca().set_xlim(0,10500) # adjust upper limit based on the longest bar (=median price per m2 per postal code) on your plot
plt.show()


####### --------- Make two maps: one showing price for the 10 least and most municipalities in Belgium,
####### another all municipalities

#### ******************************* Map 1: Most/least 10 expensive postcodes 

#for cat in ["kot", "villa"]: # use this line instead of the line below if you want to show only certain subtypes
for cat in ["any"]:    

    #df_filtered = df[df["subtype"].isin([cat])] # use this line instead of the line below if you want to show only certain subtypes
    
    df_filtered= dfj[dfj["postcode"].isin([8300,3000,8301,1348,2000,1050,8620,4623,8670,1180, # top 10 in Belgium
                                            9771,6984,6596,6590,6464,6853,6463,7340,6838,4263])].groupby('postcode').agg(  # bottom 10 in Belgium
    municipality=("Municipality",custom_aggregation_funcion), # new column!  
    median_price_pm2=("price_per_m2", "median"),
    mean_price_pm2=("price_per_m2", "mean"),
    median_price=("price", "median"), 
    mean_price=("price", "mean")
)

    postcode_shapes = gpd.read_file('./Data/georef-belgium-postal-codes/georef-belgium-postal-codes.shp')
    postcode_shapes['postcode'] = postcode_shapes['postcode'].astype(int)
    merged = postcode_shapes.merge(df_filtered, on='postcode', how='left')

    ax=merged.boundary.plot(color='gray', linewidth=0.2)
    merged.plot(ax=ax,column='median_price_pm2', legend=True, cmap='gnuplot', norm=plt.Normalize(vmin=500, vmax=11500))  
    plt.title(f"Top 10 most/least expensive municiplalities \n  based on median price per m2 \n", size=11, style='italic')       
#plt(show)

#### ******************************* Map 2: all postcodes of Belgium

#for cat in ["kot", "villa"]: # use this line instead of the line below if you want to show only certain subtypes
for cat in ["any"]:    

    #df_filtered = df[df["subtype"].isin([cat])] # use this line instead of the line below if you want to show only certain subtypes

    df_filtered= dfj.groupby('postcode').agg(  # bottom 10 in Belgium
    municipality=("Municipality",custom_aggregation_funcion), # new column!  
    median_price_pm2=("price_per_m2", "median"),
    mean_price_pm2=("price_per_m2", "mean"),
    median_price=("price", "median"), 
    mean_price=("price", "mean")
)

    postcode_shapes = gpd.read_file('./Data/georef-belgium-postal-codes/georef-belgium-postal-codes.shp')
    postcode_shapes['postcode'] = postcode_shapes['postcode'].astype(int)
    merged = postcode_shapes.merge(df_filtered, on='postcode', how='left')

    ax=merged.boundary.plot(color='gray', linewidth=0.2)
    merged.plot(ax=ax,column='median_price_pm2', legend=True, cmap='gnuplot', norm=plt.Normalize(vmin=500, vmax=11500))
    plt.title(f"All municiplalities in Belgium \n  based on median price per m2 \n", size=11, style='italic')
plt.show()

