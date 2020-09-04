#!/usr/bin/env python
# coding: utf-8

# # Predicting Survival of Titanic Passengers with Machine Learning
# by Nicholas Archambault
# 
# The Titanic disaster is one of the most indelible tragedies in human history. Its mysteries are still being unraveled today, and new techniques in data science can help shed light on certain details of the factors responsible for the ship's tragic sinking.
# 
# This project attempts to predict whether certain passengers survived the wreck, and submits the results to a Kaggle competition. We'll use data cleaning and hyperparameter selection to identify the best machine learning model for predicting passenger survival before uploading the final product to the Kaggle portal.

# Columns for this dataset include:
#    * `PassengerID` - The unique ID number of each passenger
#    * `Pclass` - The class of the passenger's ticket
#    * `Name` - The passenger's name
#    * `Sex` - The passenger's sex (male or female)
#    * `Age` - The passenger's age, as a float
#    * `SibSp` - The number of siblings and spouses the passenger had on board
#    * `Parch` - The number of parents and children the passenger had on board
#    * `Ticket` - The passenger's unique ticket number
#    * `Fare` - The fare paid, in dollars
#    * `Cabin` - The passenger's cabin number
#    * `Embarked` - The passenger's port of origin (S for Southampton, England; Q for Queenstown, Ireland; C for Cherbourg, France)
#     
#    * `Survived` - a binary variable denoting whether the passenger survived the wreck. Our target variable

# ## Cleaning Data
# 
# We can start by importing the necessary libraries and reading in prespecified train and test datasets from the Kaggle website. Our target column, `Survived`, has been eliminated from the test, or 'holdout', dataset so that it can be predicted by our machine learning model.

# In[1]:


# Import packages
import warnings
warnings.simplefilter(action ='ignore', category = FutureWarning)
warnings.simplefilter(action ='ignore', category = DeprecationWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Read in preloaded Kaggle training and testing datasets
train = pd.read_csv("train.csv")
holdout = pd.read_csv("test.csv")

holdout.head(5)


# In[3]:


train.head(5)


# In[4]:


# Get a sense of data shape
print(holdout.shape)
print(train.shape)


# ## Preprocess Data
# 
# We create a series of functions to process individual aspects of the datasets, including missing values and individual columns. These preprocessed features will be prepped and ready for use in our various models.
# 
# The functions and their descriptions are found below.

# In[5]:


# %load functions.py
def process_missing(df):
    """Handle various missing values from the data set

    Usage
    ------

    holdout = process_missing(holdout)
    """
    df["Fare"] = df["Fare"].fillna(train["Fare"].mean())
    df["Embarked"] = df["Embarked"].fillna("S")
    return df

def process_age(df):
    """Process the Age column into pre-defined 'bins' 

    Usage
    ------

    train = process_age(train)
    """
    df["Age"] = df["Age"].fillna(-0.5)
    cut_points = [-1,0,5,12,18,35,60,100]
    label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

def process_fare(df):
    """Process the Fare column into pre-defined 'bins' 

    Usage
    ------

    train = process_fare(train)
    """
    cut_points = [-1,12,50,100,1000]
    label_names = ["0-12","12-50","50-100","100+"]
    df["Fare_categories"] = pd.cut(df["Fare"],cut_points,labels=label_names)
    return df

def process_cabin(df):
    """Process the Cabin column into pre-defined 'bins' 

    Usage
    ------

    train process_cabin(train)
    """
    df["Cabin_type"] = df["Cabin"].str[0]
    df["Cabin_type"] = df["Cabin_type"].fillna("Unknown")
    df = df.drop('Cabin',axis=1)
    return df

def process_titles(df):
    """Extract and categorize the title from the name column 

    Usage
    ------

    train = process_titles(train)
    """
    titles = {
        "Mr" :         "Mr",
        "Mme":         "Mrs",
        "Ms":          "Mrs",
        "Mrs" :        "Mrs",
        "Master" :     "Master",
        "Mlle":        "Miss",
        "Miss" :       "Miss",
        "Capt":        "Officer",
        "Col":         "Officer",
        "Major":       "Officer",
        "Dr":          "Officer",
        "Rev":         "Officer",
        "Jonkheer":    "Royalty",
        "Don":         "Royalty",
        "Sir" :        "Royalty",
        "Countess":    "Royalty",
        "Dona":        "Royalty",
        "Lady" :       "Royalty"
    }
    extracted_titles = df["Name"].str.extract(' ([A-Za-z]+)\.',expand=False)
    df["Title"] = extracted_titles.map(titles)
    return df

def create_dummies(df,column_name):
    """Create Dummy Columns (One Hot Encoding) from a single Column

    Usage
    ------

    train = create_dummies(train,"Age")
    """
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df


# With the functions created, we can implement them to render cleaned datasets ready for training and testing.

# In[6]:


def cleaner(df):
    df = process_missing(df)
    df = process_age(df)
    df = process_fare(df)
    df = process_titles(df)
    df = process_cabin(df)
    df = create_dummies(df, ["Age_categories", "Fare_categories", "Title", "Cabin_type", "Sex"])
    
    return df

# Recast datasets with their cleaned versions
train = cleaner(train)
holdout = cleaner(holdout)


# ## Exploring Data
# 
# We can reengineer and visualize the `Parch` and `SibSp` columns to render them more useful for manipulation. 
# 
# The `SibSp` column shows the number of siblings and/or spouses each passenger had on board, while the `Parch` columns shows the number of parents or children each passenger had onboard. Neither column has any missing values.
# 
# These columns can be combined into a `familysize` column accounting for the total number of family members of each passenger on board.

# In[7]:


explore = train[["Parch", "SibSp", "Survived"]].copy()
explore.info()


# In[8]:


explore["familysize"] = explore[["Parch", "SibSp"]].sum(axis=1)


# In[9]:


# Visualize pivot table for probability of survival based on number of relatives aboard
for col in explore.columns.drop("Survived"):
    pivot = explore.pivot_table(index = col, values="Survived")
    pivot.plot.bar(ylim=(0,1), yticks = np.arange(0,1,.1))
    plt.show()


# The above graphs show the probability of survival for various values of `SibSp` and `Parch`. Each distribution is right skewed, dropping off as values increase.
# 
# In examining the `familysize` graph, we observe that few of the passengers with no family members aboard survived, while likelihood of survival was substantially greater for those with more family members.

# ## Engineering New Features
# 
# Based on the previous findings, we can determine a new feature: a binary category showing whether or not a passenger was alone. Below, we create the function for engineering this feature and apply it to each dataset.

# In[10]:


def solo_passenger(df):
    df["isalone"] = 0
    df["familysize"] = df[["SibSp", "Parch"]].sum(axis=1)
    df.loc[(df["familysize"] == 0), "isalone"] = 1
    df = df.drop("familysize", axis=1)
    return df

# Apply function to each dataset
train = solo_passenger(train)
holdout = solo_passenger(holdout)


# ## Selecting the Best-Performing Features
# 
# A crucial step in the machine learning workflow is feature selection. To automate this process, we can create a function using scikit Learn's `feature_selection.REFCV` class that will select best-performing features using recursive feature elimination.
# 
# Below, we create this function and apply it to the train dataset to identify the best features for prediction of the `Survived` column.

# In[11]:


# Import packages
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

def select_features(df):
    # Select only numeric features
    df = df.select_dtypes(include = [np.number]).dropna(axis=1)
    
    # Drop unnecessary and target columns
    all_X = df.drop(["PassengerId", "Survived"], axis=1)
    all_y = df["Survived"]
    
    # Instantiate random forest object
    rf = RandomForestClassifier(random_state=1)
    selector = RFECV(rf, cv=10)
    selector.fit(all_X, all_y)
    
    # Function returns printed list of the best features for rendering accurate model
    best_features = list(all_X.columns[selector.support_])
    print(best_features)
    
    return best_features

# Apply function to training dataset to select optimal features to use as predictors
train_best = select_features(train)


# In[12]:


train_best


# ## Selecting and Tuning Different Algorithms
# 
# Similar to the previous step, we can automate the process of hyperparameter optimization with a function that cycles through choices for various algorithm parameters and returns the combination with the best prediction accuracy score. Our three main model frameworks will be logistic regression, k-nearest neighbors, and random forest, meaning the function will return the best-performing candidate for each of these three model types.
# 
# The structure of the function involves a list of dictionaries. Each dictionary in the list contains a different model type, an instantiated estimator object, and a selection of hyperparameters for different attributes of that object. The function tests each combination of hyperparameters for each of the three model types, then outputs the three highest accuracy scores and hyperparameters that led to them.

# In[13]:


# Import packages
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def select_model(df, features):
    all_X = df[features]
    all_y = df["Survived"]
    
    # Dictionary of model types, objects, and parameter options
    model_dicts = [{
        "name": "LogisticRegression",
        "estimator": LogisticRegression(max_iter = 10000),
        "hyperparameters": 
            {"solver": ["newton-cg", "lbfgs", "liblinear"]}
        }, 
        {"name": "KNeighborsClassifier",
        "estimator": KNeighborsClassifier(),
        "hyperparameters": {
            "n_neighbors": range(1,20,2),
            "weights": ["distance", "uniform"],
            "algorithm": ["ball_tree", "kd_tree", "brute"],
            "p": [1,2]}
        }, 
        {"name": "RandomForestClassifier",
        "estimator": RandomForestClassifier(),
        "hyperparameters": {
            "n_estimators": [4, 6, 9],
            "criterion": ["entropy", "gini"],
            "max_depth": [2, 5, 10],
            "max_features": ["log2", "sqrt"],
            "min_samples_leaf": [1, 5, 8],
            "min_samples_split": [2, 3, 5]}
        }
    ]
    
    # Run each model type and print the most accurate version
    for i in model_dicts:
        print(i["name"])
        print("-"*len(i["name"]))
        
        grid = GridSearchCV(i["estimator"],
                           param_grid = i["hyperparameters"],
                           cv = 10)
        grid.fit(all_X, all_y)
        i["best_params"] = grid.best_params_
        i["best_score"] = grid.best_score_
        i["best_model"] = grid.best_estimator_
        
        print("Best Score: {}".format(i["best_score"]))
        print("Best Parameters: {}".format(i["best_params"]))
        
    return model_dicts


result = select_model(train, train_best)        


# ## Submitting to Kaggle
# 
# The end result of our hyperparameter and model optimization shows that the random forest algorithm is the most accurate, though its strength can be improved. If we prefer to add and re-engineer features in an effort to boost the accuracy score, we may do so and once more call the previous optimization function.
# 
# Else, we can create a function that will automatically save our best model so that it can be uploaded to Kaggle.

# In[14]:


def save_submission_file(model, features, filename = "submission.csv"):
    holdout_predictions = model.predict(holdout[features])
    submission = pd.DataFrame({"PassengerID":holdout["PassengerId"], "Survived":holdout_predictions})
    submission.to_csv(filename, index=False)


# In[15]:


save_submission_file(result[2]["best_model"], train_best)


# ## Conclusion
# 
# In this project, we stepped through an entire machine learning workflow to complete a Kaggle competition predicting whether Titanic passengers survived the shipwreck. Our best model was a random forest with ~84% prediction accuracy. This is a high mark, but there is still room for improvement. Our accuracy score can climb the Kaggle leaderboards with further feature creation and engineering or the addition of other model types from which to choose.
