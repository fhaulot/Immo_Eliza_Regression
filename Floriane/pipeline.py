import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

#Reading new file
df = pd.read_csv(r"C:\Users\fhaul\Documents\GitHub\Immo_Eliza_Regression\Dogs Data\preprocessed_data.csv")  

# Ensure 'price' is the target variable and drop it from features
X = df.drop("price", axis=1)  
y = df["price"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Create pipelines for different regression models

pipelines = {
    "Linear regression": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')), 
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ]),
    "Random Forest": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),  
        ('model', RandomForestRegressor(random_state=42))
    ]),
    "Gradient Boosting": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(random_state=42))
    ]),
    "Polynomial Regression": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ]),
}

# Creating a list to store results
results = []

# Looping through each pipeline to fit the model and evaluate performance
for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "R²": r2
    })

# Creating the sheet to display results
results_df = pd.DataFrame(results)
print(results_df)