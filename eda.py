import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

TRAIN_CSV = "/home/safe/Documentos/Evoai/Projetos/Cripto/input/g-research-crypto-forecasting/train.csv"
ASSET_DETAILS_CSV = "/home/safe/Documentos/Evoai/Projetos/Cripto/input/g-research-crypto-forecasting/asset_details.csv"

df_train = pd.read_csv(TRAIN_CSV)
print(df_train.head())

df_asset_details = pd.read_csv(ASSET_DETAILS_CSV).sort_values("Asset_ID")
print(df_asset_details)

# Two new features from the competition tutorial

def upper_shadow(df):
    return df['High'] - np.maximum(df['Close'], df['Open'])

def lower_shadow(df):
    return np.minimum(df['Close'], df['Open']) - df['Low']

# A utility function to build features from the original df
# It works for rows to, so we can reutilize it.

def get_features(df):
    df_feat = df[['Count', 'Open', 'High', 'Low', 'Close', 'Volume', 'VWAP']].copy()
    df_feat['upper_Shadow'] = upper_shadow(df_feat)
    df_feat['lower_Shadow'] = lower_shadow(df_feat)
    df_feat["high_div_low"] = df_feat["High"] / df_feat["Low"]

    #df_feat["open_sub_close"] = df_feat["Open"] - df_feat["Close"]
    df_feat['trade']=df_feat['Close']-df_feat['Open']
    df_feat['gtrade']=df_feat['trade']/df_feat['Count']
    df_feat['shadow1']=df_feat['trade']/df_feat['Volume']

    #df_feat['shadow2']=df_feat['upper_Shadow']/df['Low']
    df_feat['shadow3']=df_feat['upper_Shadow']/df['Volume']

    #df_feat['shadow4']=df_feat['lower_Shadow']/df['High']

    df_feat['shadow5']=df_feat['lower_Shadow']/df['Volume']

    df_feat['upper_Shadow_log']=np.log(df_feat['upper_Shadow'])
    df_feat['lower_Shadow_log']=np.log(df_feat['lower_Shadow'])
    return df_feat

def log(model,X_train, X_valid, y_train, y_valid,train_split=1.0):
    if train_split > 0:
        X_train=X_train[:int(train_split*X_train.shape[0])]
        y_train=y_train[:int(train_split*y_train.shape[0])]

        pred=model.predict(X_train)
        print('Training :- ')  
        print(f'MSE : {np.mean((y_train-pred)**2)}')
        print(f'CV : {pearsonr(pred,y_train)[0]}')

    pred=model.predict(X_valid)

    print('Validation :- ')
    print(f'MSE : {np.mean((y_valid-pred)**2)}')
    print(f'CV : {pearsonr(pred,y_valid)[0]}')

def get_Xy_and_model_for_asset(df_train, asset_id):
    df = df_train[df_train["Asset_ID"] == asset_id]

    # TODO: Try different features here!

    df_proc = get_features(df)
    df_proc['y'] = df['Target']
    df_proc = df_proc.dropna(how="any")

    X = df_proc.drop("y", axis=1)
    y = df_proc["y"]
    X_train=X[:int(0.7*X.shape[0])]
    y_train=y[:int(0.7*y.shape[0])]#
    X_test=X[int(X.shape[0]*0.7):]
    y_test=y[int(y.shape[0]*0.7):]

    # TODO: Try different models here!

    model = LGBMRegressor(n_estimators=200,num_leaves=300,learning_rate=0.09)
    model.fit(X_train, y_train)
    print('[Finished Training] evaluating')
    log(model,X_train, X_test, y_train, y_test,0.3)


    return X, y, model

Xs = {}
ys = {}
models = {}

for asset_id, asset_name in zip(df_asset_details['Asset_ID'], df_asset_details['Asset_Name']):
    print(f"Training model for {asset_name:<16} (ID={asset_id:<2})")
    X, y, model = get_Xy_and_model_for_asset(df_train, asset_id)    
    Xs[asset_id], ys[asset_id], models[asset_id] = X, y, model

