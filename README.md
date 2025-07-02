# Please see README.md in the main brunch for the details of pre-processing and the choice of four ML models.

## Execution.

All code is in Processor_Nad.py. You uncomment desirable model in the main method (or you can launch all of them at once):

processor.process()
#processor.fitLinearRegr()
#processor.fitRandomForest()
#processor.fitGradientBoosting()
#processor.fitPolynomialRegression()

For each model a scatter plot will be produced, and the R2 and MAE values will be printed on-screen separately for training and testing sets.

## Notes.

In my particular case, the Gradient Boost model took too long to execute, hence there no scatter plots
or values of R2 and MAE have been produced for this model.

