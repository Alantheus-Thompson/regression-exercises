from env import get_db_url
import pandas as pd
from pydataset import data
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.neighbors import KNeighborsClassifier

from scipy.stats import pearsonr
from scipy.stats import spearmanr

import warnings
warnings.filterwarnings("ignore")

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
    
    df.county=df.county.map({6037: 'LA', 6059: 'Orange', 6111: 'Ventura'})
    
    # dtype changes
    df['year built'] = df['year built'].astype(int)
    df.bedrooms = df.bedrooms.astype(int)
    
    # drop 0 bedrooms 0 bathrooms
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

def visualize_scaler(scaler, df, columns_to_scale, bins=10):
    '''
    
    Tool to visualize data before and after scaling
    
    '''
    fig, axs = plt.subplots(len(columns_to_scale), 2, figsize=(16,9))
    df_scaled = df.copy()
    df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    for (ax1, ax2), col in zip(axs, columns_to_scale):
        ax1.hist(df[col], bins=bins)
        ax1.set(title=f'{col} before scaling', xlabel=col, ylabel='count')
        ax2.hist(df_scaled[col], bins=bins)
        ax2.set(title=f'{col} after scaling with {scaler.__class__.__name__}', xlabel=col, ylabel='count')
    plt.tight_layout()
    
def scale_data(train, 
               validate, 
               test, 
               columns_to_scale=['bedrooms', 'bathrooms', 'tax_amount', 'sq_feet'],
               return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #     make the thing
    scaler = MinMaxScaler()
    #     fit the thing
    scaler.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled