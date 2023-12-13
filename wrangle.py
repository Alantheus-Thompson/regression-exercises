from env import get_db_url
import pandas as pd
from pydataset import data
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

def get_zillow():
    '''
    
    This function acquires zillow data from Codeup MySQL
    
    '''
    filename = "zillow.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)
    else:
        url = env.get_db_url('zillow')
        
        df = pd.read_sql(('''select bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt,
                                 taxamount, fips
                                 from properties_2017
                                 left join propertylandusetype
                                     using (propertylandusetypeid)
                                 WHERE propertylandusedesc = ("Single Family Residential")'''), url)
        df.to_csv(filename)
    return df


def prep_zillow(df):
    '''
    
    prepares zillow data by changing column names, a few dtypes, dropping nulls, and houses with 0 bedrooms and 0 bathrooms.
    
    '''
    column_name_changes = {'bedroomcnt': 'bedrooms', 'bathroomcnt': 'bathrooms', 'calculatedfinishedsquarefeet': 
                       'sqft', 'taxvaluedollarcnt': 'appraisal', 'yearbuilt': 'year built', 'taxamount': 'taxes',
                      'fips': 'county'}
    df.rename(columns=column_name_changes, inplace=True)
    
    df = df.dropna()
    
    # dtype changes
    df['year built'] = df['year built'].astype(int)
    df.bedrooms = df.bedrooms.astype(int)
    df.county = df.county.astype(object)
    
    df.county=df.county.map({6037: 'LA', 6059: 'Orange', 6111: 'Ventura'})
    
    df = df[(df['bedrooms'] != 0) & (df['bathrooms'] != 0)]
    
    return df



def split_data(df):
    '''
    
    Splits df into train, validate, & split (60, 20, 20 split)
   
    '''
    train, validate_test = train_test_split(df,
                                            train_size=0.60,
                                            random_state=123,
                                            )

 
    validate, test = train_test_split(validate_test,
                                      test_size=0.50,
                                      random_state=123,
                                      )

    return train, validate, test

def wrangle_function():
    '''
    
    Combines acquire, prepare, and split functions into one callable function.
    
    '''
    
    train, validate, test=split_data(prep_zillow(get_zillow()))
    
    return train, validate, test
