# Predicting Survival of Titanic Passengers with Machine Learning
by Nicholas Archambault

The Titanic disaster is one of the most indelible tragedies in human history. Its mysteries are still being unraveled today, and new techniques in data science can help shed light on certain details of the factors responsible for the ship's tragic sinking.

This project attempts to predict whether certain passengers survived the wreck, and submits the results to a Kaggle competition. With data cleaning and hyperparameter selection, the optimal model for predicting passenger survival can be identified.

## Goals
1. Clean, re-engineer, and preprocess features of the dataset in order to make them suitable for machine learning.
2. Automate the identification of the combination of features that will render the best model.
3. Create a function that automatically tests various model types and hyperparameter combinations.
4. Submit best model result to Kaggle forum.

## Output
A comparison of logistic regression, decision tree, and random forest models with optimized hyperparameters. Best model runs a random forest and accurately predicts survival of ~84% of passengers, with substantial room for continued feature engineering and score improvement.