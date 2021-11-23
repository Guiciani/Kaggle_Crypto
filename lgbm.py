import os
import random
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
#import gresearch_crypto
import pdb; pdb.set_trace()


TRAIN_CSV = pd.read_csv("../home/safe/Documentos/Evoai/Projetos/Cripto/input/g-research-crypto-forecasting/train.csv")
ASSET_DETAILS_CSV = pd.read_csv("../home/safe/Documentos/Evoai/Projetos/Cripto/input/g-research-crypto-forecasting/asset_details.csv")

SEED = 2021

REMOVE_LB_TEST_OVERLAPPING_DATA = True


def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

fix_all_seeds(SEED)

df_train = pd.read_csv(TRAIN_CSV)
print(df_train.head())

# Remove the future
if REMOVE_LB_TEST_OVERLAPPING_DATA:
    df_train['datetime'] = pd.to_datetime(df_train['timestamp'], unit='s')
    df_train = df_train[df_train['datetime'] < '2021-06-13 00:00:00']

df_asset_details = pd.read_csv(ASSET_DETAILS_CSV).sort_values("Asset_ID")
print(df_asset_details)

#TREINAMENTO

# Duas novas Features
def upper_shadow(df):
    return df['High'] - np.maximum(df['Close'], df['Open'])

def lower_shadow(df):
    return np.minimum(df['Close'], df['Open']) - df['Low']

# Uma função para construir as features do DF original
# Funciona tambem para linhas.
def get_features(df):
    df_feat = df[['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']].copy()
    df_feat['Upper_Shadow'] = upper_shadow(df_feat)
    df_feat['Lower_Shadow'] = lower_shadow(df_feat)
    return df_feat

def get_Xy_and_model_for_asset(df_train, asset_id):
    df = df_train[df_train["Asset_ID"] == asset_id]
    
    # TODO: Try different features here!
    df_proc = get_features(df)
    df_proc['y'] = df['Target']
    df_proc = df_proc.dropna(how="any")
    
    X = df_proc.drop("y", axis=1)
    y = df_proc["y"]
    
    # TODO: Try different models here!
    model = LGBMRegressor(n_estimators=10)
    model.fit(X, y)
    return X, y, model

# Loop em cima dos ativos

Xs = {}
ys = {}
models = {}

for asset_id, asset_name in zip(df_asset_details['Asset_ID'], df_asset_details['Asset_Name']):
    print(f"Training model for {asset_name:<16} (ID={asset_id:<2})")
    X, y, model = get_Xy_and_model_for_asset(df_train, asset_id)    
    Xs[asset_id], ys[asset_id], models[asset_id] = X, y, model

# Checando a interface do modelo
x = get_features(df_train.iloc[1])
y_pred = models[0].predict([x])
print(y_pred[0])

all_df_test = []

env = gresearch_crypto.make_env()
iter_test = env.iter_test()

for i, (df_test, df_pred) in enumerate(iter_test):
    for j , row in df_test.iterrows():
        
        model = models[row['Asset_ID']]
        x_test = get_features(row)
        y_pred = model.predict([x_test])[0]
        
        df_pred.loc[df_pred['row_id'] == row['row_id'], 'Target'] = y_pred
        
        
        # Print just one sample row to get a feeling of what it looks like
        if i == 0 and j == 0:
            display(x_test)

    # Display the first prediction dataframe
    if i == 0:
        display(df_pred)
    all_df_test.append(df_test)

    # Send submissions
    env.predict(df_pred)

df_test = pd.concat(all_df_test)
df_test['datetime'] = pd.to_datetime(df_test['timestamp'], unit='s')
df_train['datetime'] = pd.to_datetime(df_train['timestamp'], unit='s')

print(df_train['datetime'].max())

print(df_test['datetime'].min())