from env import get_db_url
import pandas as pd
from pydataset import data
import os

def get_titanic_data():
    filename = "titanic.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)
    else:
        url = get_db_url('titanic_db')
        df_titanic = pd.read_sql(('SELECT * FROM passengers'), url)
        df_titanic.to_csv(filename)
    return df_titanic

def get_iris_data():
    filename = "iris.csv"
    
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)
    
    else:
        url=get_db_url('iris_db')
        df_iris = pd.read_sql(('''
        select *
        from measurements
        join species
            using (species_id)'''),url)
        df_iris.to_csv(filename)
    return df_iris

def get_telco_data():
    filename = "churn.csv"
        
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)

    else:
        url = get_db_url('telco_churn')
        df_telco_churn = pd.read_sql(('''
        SELECT * From customers
        join contract_types
            using (contract_type_id)
        join internet_service_types
            using(internet_service_type_id)
        join payment_types
            using(payment_type_id)'''), url)
        df_telco_churn.to_csv(filename)
    return df_telco_churn     
  

