import pandas as pd
import os
from pydataset import data
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import env
from sklearn.model_selection import train_test_split
import acquire


def prep_iris():
    '''
    
    '''
    df_iris=acquire.get_iris_data()
    df_iris=df_iris.drop(columns=['species_id','measurement_id'])
    df_iris=df_iris.rename(columns={'species_name':'species'})
    return df_iris

def prep_titanic():
    '''
    
    '''
    df_titanic=acquire.get_titanic_data()
    df_titanic=df_titanic.drop(columns=['embarked', 'age','deck', 'class'])
    df_titanic.pclass = df_titanic.pclass.astype(object)
    df_titanic.embark_town = df_titanic.embark_town.fillna('Southampton')
    
    return df_titanic

def prep_telco():
    '''
    
    '''
    df_telco_churn = acquire.get_telco_data()
    df_telco_churn=df_telco_churn.drop(columns=['payment_type_id', 'internet_service_type_id','contract_type_id'],errors='ignore')
    df_telco_churn=df_telco_churn.isnull().fillna('No Internet')
    return df_telco_churn


def split_data(df, target_column):
    '''
   
    '''
    train, validate_test = train_test_split(df,
                                            train_size=0.60,
                                            random_state=123,
                                            stratify=df[target_column])

 
    validate, test = train_test_split(validate_test,
                                      test_size=0.50,
                                      random_state=123,
                                      stratify=validate_test[target_column])

    return train, validate, test

