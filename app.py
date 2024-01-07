import streamlit as st 
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.model_selection import GridSearchCV
import re
from collections import Counter

st.write("")

# Helper Functions for transformations and pipelines

# Pipe data processor (Feature Engine)
def process_data_pipe(data, pipeline_functions, prediction_col):
    """Process the data for a guided model."""
    for function, arguments, keyword_arguments in pipeline_functions:
        if keyword_arguments and (not arguments):
            data = data.pipe(function, **keyword_arguments)
        elif (not keyword_arguments) and (arguments):
            data = data.pipe(function, *arguments)
        else:
            data = data.pipe(function)
    X = data.drop(columns=[prediction_col])
    Y = data.loc[:, prediction_col]
    return X, Y


# Pipe data processor for test/validation set (Feature Engine)
def process_data_pipe_test(data, pipeline_functions):
    """Process the data for a guided model."""
    for function, arguments, keyword_arguments in pipeline_functions:
        if keyword_arguments and (not arguments):
            data = data.pipe(function, **keyword_arguments)
        elif (not keyword_arguments) and (arguments):
            data = data.pipe(function, *arguments)
        else:
            data = data.pipe(function)
    return data


# Adds new column to indicate whether the email is a reply or not.
def if_reply(data):
    """
    Input:
    data (DataFrame): a DataFrame containing at least the Subject column.
    
    Output:
    a DataFrame with a new column "IsReply" containing 0 for No and 1 for Yes.
    """
    data_new = data.copy()
    reply_rx = r'^Subject:\s*RE:'
    data_new['IsReply'] = (data['subject'].str.contains(reply_rx, case=False, na=False)).astype(int)
    return data_new


# Adds columns to indicate number of exclamation marks in each email.
def add_exclamation_count(data):
    """
    Adds a new column to the DataFrame that contains the number of exclamation marks in each email.

    Input:
    data (DataFrame): a DataFrame containing at least the Email column.

    Output:
    DataFrame with a new column "NumExp" containing the count of "!" in the email column.
    """
    data_new = data.copy()
    data_new['NumExp'] = data['email'].str.count('!')

    mean_num_exp = data_new['NumExp'].mean()
    std_num_exp = data_new['NumExp'].std()

    # Standardize the 'NumExp' feature
    data_new['NumExp_S'] = data_new['NumExp'].apply(lambda x: (x - mean_num_exp) / std_num_exp)
    return data_new


# Modified Words in Texts to accomodate pipelining
def words_in_texts_new(df):
    """
    For each word in 'words', add a new column to 'df' that indicates the presence of the word in the 'email' column.
    
    Args:
    words (list): Words to find.
    df (DataFrame): DataFrame containing the 'email' column.
    
    Returns:
    DataFrame with new columns, each one representing one of the words in 'words'.
    """
    data = df.copy()
    words = ['free', 'credit', 'offer', 'guaranteed', 'money', 'subscribe', '100%', 'low', 'click', 'please', 'html']
    for word in words:
        # Column names are the words prefixed with some identifier if needed
        col_name = f"w_{word}"
        # Indicate presence of word with 1, absence with 0
        data[col_name] = data['email'].str.contains(r'\b' + word + r'\b', case=False, na=False).astype(int)
    return data


# Select Columns
def select_columns(data, *columns):
    """Select only columns passed as arguments."""
    return data.loc[:, columns]

def num_words_in_body(df):
    df['body_word_count'] = df['email'].apply(lambda x: len(str(x).split()))
    mean_words_body = df['body_word_count'].mean()
    std_words_body = df['body_word_count'].std()
    df['body_word_count_standardized'] = df['body_word_count'].apply(lambda x: (x - mean_words_body) / std_words_body)
    return df


def ProcessLr(data, is_test_set=False):
    if not is_test_set:
        # Processing for the training set
        # Pipeline
        train_pipelines = [ 
            (words_in_texts_new, None, None),
            (if_reply, None, None),
            (add_exclamation_count, None, None),
            (num_words_in_body, None, None),
            (select_columns, [
                'w_free',
                'w_credit',
                'w_offer',
                'w_guaranteed',
                'w_money',
                'w_subscribe',
                'w_100%',
                'w_low',
                'w_click',
                'w_please',
                'w_html',
                'NumExp_S',
                'spam',
                'body_word_count_standardized',
                'IsReply'
            ], None)
        ]

        # Applying Pipeline
        X_train, Y_train = process_data_pipe(data, train_pipelines, 'spam')
    
    else:
        # Processing for the test set
        
        # Pipeline
        test_pipelines = [ 
            (words_in_texts_new, None, None),
            (if_reply, None, None),
            (add_exclamation_count, None, None),
            (num_words_in_body, None, None),
            (select_columns, [
                'w_free',
                'w_credit',
                'w_offer',
                'w_guaranteed',
                'w_money',
                'w_subscribe',
                'w_100%',
                'w_low',
                'w_click',
                'w_please',
                'w_html',
                'NumExp_S',
                'body_word_count_standardized',
                'IsReply'
            ], None)
        ]
        # Applying Pipeline
        X_test = process_data_pipe_test(data, test_pipelines)

    # Return predictors (X) and response (Y) variables separately
    if is_test_set:
        # Predictors 
        X = X_test
        return X
    else:
        # Predictors
        X = X_train
        # Response variable
        Y = Y_train
        
        return X, Y

# Model
my_model = LogisticRegression()
# X, Y = ProcessLr(train, is_test_set=False)
# , Y)
# Model Predictions
# # Y_pred = my_model.predict(X)
# training_accuracy = np.mean(Y_pred == Y)
# print("Training Accuracy: ", training_accuracy)