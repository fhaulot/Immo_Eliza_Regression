# Immo_Eliza_Regression

## Description

The application tests 4 machine learning algorithms to predict
property prices in Belgium based on scraping of one on-line immo agency.
The input data is contained in a csv file (Floriane\cleaned_data.csv).
In the process of creating models, the sample is split in two parts : the training and the verification/testing part (using 4:1 ratio).
Each model is trained (calibrated) using the training set.
The solution is then applied to the testing set to predict the prices of both sets.
The quality of the model is evaluated in three ways:
by looking at the scatter plot between predicted and actual prices;
by plotting the histogram of the price difference;
by calculating two standard statistical values that evaluate the average price difference: R2 and MAE.

## Installation

A python environment is used. 
Also libraries : 
- Pandas
- Sklearn
- Matpotlib
- Numpy
- Seaborn

You may find the environment in the requirements.txt file

## Usage

- Running Preprocessed.py to eliminate Nan and set the parameters
- Running Main.py to apply the preprocessing on the cleaned_data.csv file
- Running pipeline.py to predict price with different models

If you want to explore the database we used for this project, you will find it in the Dogs Data folder with the scripts that were used to create it.


## Pre-processing procedure

After reading csv file we discard the following parameters/columns:
ID, URL, Locality, Municipality, Price per square meter.
These parameters have no value in regard to the ML techniques,
since they are either strings duplicated in other variables or
depend on the predicted variable (price per m2 depends was calculated using total price).
Next, we converted the following parameters to a sequence of numbers:
Building condition and EPC score, thus treating these variables as categorial ordered parameters.
Logical parameters: Has terrace, Has parking have been converted to 0 (=No) or 1 (=Yes).

Next we identified 3 groups of parameters that are related to each other:
1) Has garden, Garden surface,
2) Type and Subtype
3) Postcode, Region, Province.
Only one parameter should be chosen for model training from each group.
As shown in put presentation, we evaluated how much each parameter improves the fit
based on the linear regression model, and retained the best one in each group.
Postcodes have been converted to latitude and longitude using a table we found on the internet.

The resulting parameters used in our 4 models are:
Garden surface, Ubtype, Province, Bedroom count, Habitable surface,
Has terrace, Has parking, Building condition, EPC score.

## Slides description
In the Kriek_Immo_eliza_regression.pdf. This is the presentation we made in front of the group. This is useful to have a better visualisation with the plots we have made. 

The first slide outlines the progressive construction of a linear regression model by evaluating which variables to include or exclude and how their inclusion affects model performance.

The second slide compares three regression models by plotting actual prices (x-axis) vs predicted prices (y-axis), using red for training data and blue for test data. The black diagonal line represents perfect predictions.

The third slide presents the performance of a Gradient Boosting regression model, showing it delivers high accuracy and balance between training and test sets, with predictions closely aligned to actual property prices making it the best-performing model so far

The fourth compares four regression models : Linear Regression, Random Forest, Gradient Boosting, and Polynomial Regression : based on test/train performance using MAE and R² metrics.

The fifth slide is a boxplot that visualizes the prediction errors (residues) for each subtype of property after fitting a polynomial regression model. In the last slide, we used polynomial regression than gradient boosting to plot residuals for each property subtype because the last one take too many time to run
