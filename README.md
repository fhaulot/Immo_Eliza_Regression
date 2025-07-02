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
Pandas
Sklearn
Matpotlib
Numpy
Seaborn 

## Usage

Running Preprocessed.py to eliminate Nan and set the parameters
Running Main.py to apply the preprocessing on the cleaned_data.csv file
Running pipeline.py to predict price with different models


### Pre-processing procedure

After reading csv file we discard following parameters/columns:
