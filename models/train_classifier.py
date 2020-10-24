import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

def load_data(database_filepath):
    '''
    Function to load database
    Input: Database file path
    Output: Features X and Target y dataframe for machine learning
    '''
    engine = create_engine('sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_messages', con= engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    return X, y


def tokenize(text):
    '''
    Function to clean the text
    Input: text
    Output: clean text
    '''
    tokens = word_tokenize(text)
    lem = WordNetLemmatizer()
    process_tokens = []
    for token in tokens:
        process_token = lem.lemmatize(token).lower().strip()
        process_tokens.append(process_token)
    return process_tokens


def build_model():
    '''
    Function to build the model
    Input: None
    Output: Model 
    '''
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer= tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])
    parameters = {
    'clf__estimator__criterion': ['entropy']
}

    cv = GridSearchCV(pipeline, param_grid= parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function to evaluate the model
    Input: Model, X_test, y_test, category_names
    Output: Classification report with f1, recall, precision score
    '''
    y_pred = model.predict(X_test)
    for i, column_name in enumerate(y_test):
        print (column_name)
        print (classification_report(y_test[column_name], y_pred[:, i]))
    pass


def save_model(model, model_filepath):
    '''
    Function to save the model
    Input: model, model_path
    Output: saved the model
    '''
    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()