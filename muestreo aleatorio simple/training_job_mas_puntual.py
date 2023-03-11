import argparse
import cgi
import json
import os
import subprocess
import sys
from io import StringIO 
import joblib
import numpy as np
import pandas as pd
from sagemaker_containers.beta.framework import encoders, worker
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler



subprocess.check_call(
    [sys.executable, '-m', 'pip', 'install', 'awswrangler==2.14.0'])

subprocess.check_call(
    [sys.executable, '-m', 'pip', 'install', 'pandas==1.1.5'])

subprocess.check_call(
    [sys.executable, '-m', 'pip', 'install', 'category-encoders==2.2.2'])

subprocess.check_call(
    [sys.executable, '-m', 'pip', 'install', 'matplotlib==3.1'])

subprocess.check_call(
    [sys.executable, '-m', 'pip', 'install', 'seaborn==0.9.0'])

subprocess.check_call(
    [sys.executable, '-m', 'pip', 'install', 'scorecardpy==0.1.9.2'])

subprocess.check_call(
    [sys.executable, '-m', 'pip', 'install', 'imbalanced-learn==0.8.1'])




import seaborn as sns
import matplotlib.pyplot as plt
import scorecardpy as sc
import awswrangler as wr
import pandas as pd


#import awswrangler as wr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
plt.rcParams["figure.figsize"] =8,4 # para ajustar el tamaño de imagen
import warnings
warnings.filterwarnings("ignore") #para evitar errores de operaciones matematicas
import random
import statistics
import math
#from pandas_profiling import ProfileReport 




ruta_s3 = 's3://data-tecnicas-muestreo/datasets/Telco_customer_churn.csv'   # Ruta S3 donde esta el dataset 
    
dataset = wr.s3.read_csv(path = ruta_s3, sep = ',')
    
target = dataset['tenure']    # Variable de interes, estimacion puntual
    
def mas_puntual(target, LEE):
    
     random.seed(123) # fijar semilla valores aleatorios
    
     mean_poblacional = target.mean()   # promedio poblacional (variable de interes)
     N = len(target) 
     D =  pow(LEE, 2)/4    # limite margenes error estimacion poblacional
     var = statistics.variance(target)   # Varianza Poblacional
     n =  math.floor((N*var)/((N-1)*D + var)) # seleccion de muestra aleatoria
    
    
     index_sample = random.sample (list(target.index), n)  # muestreo aleatorio simple
     target_sample = target[index_sample]  # seleccion muestra por index
    
     mean_est = target_sample.mean()  # promedio muestral estimado (variable de interes)
    
     var_mean_est = (np.var(target_sample)/n)*(1-(n/N))  # varianza muestral estimada (variable de interes)
     error_est =  2*math.sqrt(var_mean_est)  # error de estimacion muestra estimada (variable de interes)

     LI =  mean_est - error_est  # limite superior estimacion muestral
     LS =  mean_est + error_est  # limite inferior estimacion muestral
    
     data_filtrada = dataset.iloc[target_sample.index, :]  # dataset muestral
    
     print("El intervalo de confianza de la proporcion esta entre", "[",LI, LS,"]")
     print("tamaño total poblacional:" , len(dataset) )
     print("tamaño de muestra estimada:" , len(data_filtrada) )
     print("promedio poblacional estimado:" , mean_poblacional )
     print("promedio muestral estimado:" , mean_est )
     print("error de estimacion:" , error_est )
    
     return(data_filtrada)

    
df_1 = mas_puntual(target, 0.07)
    
    
df_path_s3='s3://data-tecnicas-muestreo/datasets/Telco_customer_churn_mas_puntual.csv'    # Ubicacion en S3 donde se va almacenar el dataset muestral

wr.s3.to_csv(df=df_1, path=df_path_s3, index=False)



def input_fn(input_data, content_type):
    """Parse input data payload.

    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """
    # cgi.parse_header extracts all arguments after ';' as key-value pairs
    # e.g. cgi.parse_header('text/csv;label_size=1;charset=utf8') returns
    # the tuple ('text/csv', {'label_size': '1', 'charset': 'utf8'})
    content_type, params = cgi.parse_header(content_type.lower())

    if content_type == 'text/csv':
        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data), header=None, nrows=0)

        if len(df.columns) == len(feature_columns_names) + 1:
            # This is a labelled example, includes the label column
            df = pd.read_csv(
                StringIO(input_data),
                names=feature_columns_names + [label_column],
                dtype=merge_two_dicts(
                    feature_columns_dtype, label_column_dtype))

        elif len(df.columns) == len(feature_columns_names):
            # This is an unlabelled example
            df = pd.read_csv(
                StringIO(input_data),
                names=feature_columns_names,
                dtype=feature_columns_dtype)

        return df
    else:
        raise NotImplementedError(
            "content_type '{}' not implemented!".format(content_type))


def predict_fn(input_data, model):
    """Preprocess input data.

    We implement this because the default predict_fn uses .predict(), but our model is a preprocessor
    so we want to use .transform().

    The output is returned in the following order:

        rest of features either one hot encoded or standardized
    """
    features_enc = model['features_enc']
    label_enc = model['label_enc']

    if label_column in input_data:
        X, y = split_X_y(input_data)
        features = features_enc.transform(X)
        encoded_label = label_enc.transform(y)
        # Return the label (as the first column) and the set of features.
        return np.insert(features, 0, encoded_label, axis=1)
    else:
        # Return only the set of features
        return features_enc.transform(input_data)


def model_fn(model_dir):
    """Deserialize fitted model."""

    return {
        'features_enc': joblib.load(os.path.join(model_dir, 'features_enc.joblib')),
        'label_enc': joblib.load(os.path.join(model_dir, 'label_enc.joblib'))
    }


def output_fn(prediction, accept):
    """Format prediction output.

    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    # cgi.parse_header extracts all arguments after ';' as key-value pairs
    # e.g. cgi.parse_header('text/csv;label_size=1;charset=utf8') returns
    # the tuple ('text/csv', {'label_size': '1', 'charset': 'utf8'})
    accept, params = cgi.parse_header(accept.lower())

    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}

        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        raise NotImplementedError(
            "accept '{}' not implemented!".format(accept))
