import gc
import numpy as np
import pandas as pd
import datatable as dt
pd.set_option("display.max_columns", None)
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt


plt.rcParams.update({'font.size': 14})
data_folder = "../home/safe/Documentos/Evoai/Projetos/Cripto/input/g-research-crypto-forecasting/"
# data_folder = "/t/Datasets/kaggle_crypto/"
# asset_details = pd.read_csv(data_folder + 'asset_details.csv', low_memory=False) 
# train = pd.read_csv(data_folder + 'train.csv', low_memory=False)
asset_details = dt.fread("/home/safe/Documentos/Evoai/Projetos/Cripto/input/g-research-crypto-forecasting/asset_details.csv").to_pandas()
train = dt.fread("/home/safe/Documentos/Evoai/Projetos/Cripto/input/g-research-crypto-forecasting/train.csv").to_pandas()

df_train = pd.read_csv("/home/safe/Documentos/Evoai/Projetos/Cripto/input/g-research-crypto-forecasting/train.csv")

rename_dict = {}

for a in asset_details['Asset_ID']: rename_dict[a] = asset_details[asset_details.Asset_ID == a].Asset_Name.values[0]
train['timestamp'] = train['timestamp'].astype('datetime64[s]')
train_daily = pd.DataFrame()

for asset_id in asset_details.Asset_ID:
    train_single = train[train.Asset_ID == asset_id].copy()
    train_single_new = train_single[['timestamp','Count']].resample('D', on='timestamp').sum()
    train_single_new['Open'] = train_single[['timestamp','Open']].resample('D', on='timestamp').first()['Open']
    train_single_new['High'] = train_single[['timestamp','High']].resample('D', on='timestamp').max()['High']
    train_single_new['Low'] = train_single[['timestamp','Low']].resample('D', on='timestamp').min()['Low']
    train_single_new['Close'] = train_single[['timestamp','Close']].resample('D', on='timestamp').last()['Close']
    train_single_new['Volume'] = train_single[['timestamp','Volume']].resample('D', on='timestamp').sum()['Volume']
    train_single_new['Asset_ID'] = asset_id
    train_daily = train_daily.append(train_single_new.reset_index(drop=False))

train_daily = train_daily.sort_values(by = ['timestamp', 'Asset_ID']).reset_index(drop=True)
train_daily = train_daily.pivot(index='timestamp', columns='Asset_ID')[['Count', 'Open', 'High', 'Low', 'Close', 'Volume']]
train_daily = train_daily.reset_index(drop=False)
train_daily['year'] = pd.DatetimeIndex(train_daily['timestamp']).year

fig = make_subplots( rows=len(asset_details.Asset_ID), cols=1, subplot_titles=(asset_details.Asset_Name) )

for i, asset_id in enumerate(asset_details.Asset_ID):
    fig.append_trace(go.Candlestick(x=train_daily.timestamp, open=train_daily[('Open', asset_id)], high=train_daily[('High', asset_id)], low=train_daily[('Low', asset_id)], close=train_daily[('Close', asset_id)]),row=i+1, col=1,)
    fig.update_xaxes(range=[train_daily.timestamp.iloc[0], train_daily.timestamp.iloc[-1]], row=i+1, col=1)

fig.update_layout(xaxis_rangeslider_visible = False, 
                  xaxis2_rangeslider_visible = False, 
                  xaxis3_rangeslider_visible = False,
                  xaxis4_rangeslider_visible = False,
                  xaxis5_rangeslider_visible = False,
                  xaxis6_rangeslider_visible = False,
                  xaxis7_rangeslider_visible = False,
                  xaxis8_rangeslider_visible = False,
                  xaxis9_rangeslider_visible = False,
                  xaxis10_rangeslider_visible = False,
                  xaxis11_rangeslider_visible = False,
                  xaxis12_rangeslider_visible = False,
                  xaxis13_rangeslider_visible = False,
                  xaxis14_rangeslider_visible = False,
                  height=3000, width=800, 
                  #title_text="Subplots with Annotations"
                      margin = dict(
        l = 0,
        r = 0,
        b = 0,
        t = 30,
        pad = 0)
                 )                
fig.show()

del train, train_daily,train_single, asset_details
gc.collect()
plt.close()