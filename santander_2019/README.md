# Santander Customer Transaction Prediction 

## Description

The object of this competition is to predict whether a customer will make a specific transaction again in the
future. Each data sample includes 200 (unlabeled) features to aid in making this prediction. The target is
problem is binary classification: 0 for the customer does NOT make the transaction again or 1 for when
the customer will make the transaction again.

Predictions are the probabilty (e.g. 0.25) that a customer will make the transaction again.

## Training Plan

The training data is quite large (especially since there are so many features). So we will use out
of core techniques to study the data; i.e. we will use minibatch training to create a transformer
that will reduce the data a small number of interesting features. 

After reducing the dimensionality of the data, we will be able to fit it into memory. Then we will train
a predictor on the entirety of the reduced data.

## Metric

The data contains a proportionately small number of repeat transactions (targets of class 1). So accuracy
is a poor metric to evaluate the usefulness of our model; a silly model that always predicts the customer
will not repeat (class 0) will have very good accuracy.

Instead the model is scored using the Area Under the Receiver Operator Characteristic Curve. This metric
measures the trade off between increasing the true positive rate and increasing the false positive rate. 
That is, it measures this tradeoff as you change the cut-off level to determine when you
are class 0 or class 1 (probability less than the cut-off is class 0 and similar for larger).

## Files

The main directory contains two notebooks:

* `BestDirectionsModel.ipynb` - This notebook has the training of the model, the decision of the final predictor,
and the encoding of the model into a json file. 
* `BestDirectionsModel_tests.ipynb` - This notebook decodes the JSON format of the model, double checks the
training score to make sure the decoding worked, and then makes predictions on the testing dataset. 

### data/ 

This sub-directory is where the project's data should go. For space concerns, the data isn't tracked by git.

### my\_src/ 

This sub-directory contains source files for classes that make up our model's transformers and estimators. It
also has classes to help with doing k-fold cross-validation across minibatches of our data and with using
JSON encoding/decoding for model persistence.

