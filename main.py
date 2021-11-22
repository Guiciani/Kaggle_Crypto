import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import time
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import MultiOutputRegressor

data_folder = "/home/safe/Documentos/Evoai/Projetos/Cripto/input/g-research-crypto-forecasting/"


# Carregando dados para treinamento
crypto_df = pd.read_csv(data_folder + 'train.csv')
print(crypto_df.head(10))

# Carregando dados com detalhamento das moedas
asset_details = pd.read_csv(data_folder + 'asset_details.csv')
print(asset_details)


##############################################################
# Plotando os valores do Bitcoin com o Plotly

btc = crypto_df[crypto_df['Asset_ID']==1].set_index('timestamp') # Chamando o Asset_ID 1 para trazer o Bitcoin
btc_mini = btc.iloc[-200:] #Filtrando os valores das ultimas 200 linhas

fig = go.Figure(data=[go.Candlestick(x=btc_mini.index, open = btc_mini['Open'], high = btc_mini['High'], low = btc_mini['Low'], close = btc_mini['Close'])])
fig.show()

eth = crypto_df[crypto_df['Asset_ID'] == 6].set_index('timestamp') # Asset_ID 6 para Ethereum
print(eth.info(show_counts = True))
print(eth.isna().sum())
print(btc.head())

# Verificando o Range dos dados atraves do datetime
beg_btc = btc.index[0].astype('datetime64[s]')
end_btc = btc.index[-1].astype('datetime64[s]')
beg_eth = btc.index[0].astype('datetime64[s]')
end_eth = btc.index[-1].astype('datetime64[s]')

print('BTC data goes from', beg_btc, 'to', end_btc)
print('ETH data goes from', beg_eth, 'to', end_eth)

# Verificando dados ausentes pelo datetime
print((eth.index[1:]-eth.index[:-1]).value_counts().head())


# Como existem muitos dados ausentes, precisamos preprocessar os dados
# em um formato sem buraco de tempo com o .reindex() preenchendo o campo
# ausente com o valor anterior.

eth = eth.reindex(range(eth.index[0], eth.index[-1]+60,60), method = 'pad')
print((eth.index[1:]-eth.index[:-1]).value_counts().head())

# Até esse ponto importei os dados e organizei o data set pelo datetime
# atraves do index

# Data Visualisation

# plot vwap time series para os 2 ativos

f = plt.figure(figsize = (15,4))

# Ajustando o range do BTC também (mesmo esquema do ethereum) linha 50
btc = btc.reindex(range(btc.index[0], btc.index[-1] + 60,60), method = 'pad')

ax = f.add_subplot(121)
plt.plot(btc['Close'], label = 'BTC')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Bitcoin')

ax2 = f.add_subplot(122)
ax2.plot(eth['Close'], color = 'red', label = 'ETH')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Ethereum')

plt.tight_layout()
print(plt.show())

# Ainda que tenham historias completamente diferente, os ativos podem se
# correlacionar de algumas formas, observamos abaixo 

# Função auxiliar, from datetime to timestamp
totimestamp = lambda s: np.int32(time.mktime(datetime.strptime(s, '%d/%m/%Y').timetuple()))

# Criando intervalos

btc_mini_2021 = btc.loc[totimestamp('01/06/2021'):totimestamp('01/07/2021')]
eth_mini_2021 = eth.loc[totimestamp('01/06/2021'):totimestamp('01/07/2021')]

# Plotando timeseries para os ativos

f = plt.figure(figsize=(7,8))

ax = f.add_subplot(211)
plt.plot(btc_mini_2021['Close'], label = 'btc')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Bitcoin Close')

ax2 = f.add_subplot(212)
ax2.plot(eth_mini_2021['Close'], label = 'eth')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Ethereum Close')

plt.tight_layout()
plt.show()

# Definindo uma função para computar os logs

def log_return(series, periods = 1):
    return np.log(series).diff(periods = periods)

lret_btc = log_return(btc_mini_2021.Close)[1:]
lret_eth = log_return(eth_mini_2021.Close)[1:]
lret_btc.rename('lret_btc',inplace = True)
lret_eth.rename('lret_eth',inplace = True)

plt.figure(figsize=(8,4))
plt.plot(lret_btc)
plt.plot(lret_eth)
plt.show()

# Correlação entre ativos
# Utilizando o join em um unico Dataframe

lret_btc_long = log_return(btc.Close)[1:]
lret_eth_long = log_return(eth.Close)[1:]
lret_btc_long.rename('lret_btc', inplace = True)
lret_eth_long.rename('lret_eth', inplace = True)
two_assets = pd.concat([lret_btc_long, lret_eth_long],axis = 1)

# Agrupando as linhas e usando o.corr() para correlação entre colunas
corr_time = two_assets.groupby(two_assets.index//(10000*60)).corr().loc[:,'lret_btc'].loc[:,'lret_eth']

corr_time.plot()
plt.xticks([])
plt.ylabel('Correlation')
plt.title('Correlation between BTC and ETH over time');


# Como resultado, vemos uma alta mas variavel correlação. A dnamica de correlacao
# é volatil ao longo do tempo sendo assim um ambiente altamente não-estacionario


# Veriicando correlações entre todos os ativos

all_assets_2021 = pd.DataFrame([])
for asset_id, asset_name in zip(asset_details.Asset_ID, asset_details.Asset_Name):
  asset = crypto_df[crypto_df["Asset_ID"]==asset_id].set_index("timestamp")
  asset = asset.loc[totimestamp('01/01/2021'):totimestamp('01/05/2021')]
  asset = asset.reindex(range(asset.index[0],asset.index[-1]+60,60),method='pad')
  lret = log_return(asset.Close.fillna(0))[1:]
  all_assets_2021 = all_assets_2021.join(lret, rsuffix=asset_name, how="outer")

plt.imshow(all_assets_2021.corr())
plt.yticks(asset_details.Asset_ID.values, asset_details.Asset_Name.values)
plt.xticks(asset_details.Asset_ID.values, asset_details.Asset_Name.values, rotation='vertical')
plt.colorbar()


# Feature Design
upper_shadow = lambda asset: asset.High - np.maximum(asset.Close,asset.Open)
lower_shadow = lambda asset: np.minimum(asset.Close,asset.Open)- asset.Low

X_btc = pd.concat([log_return(btc.VWAP,periods=5), log_return(btc.VWAP,periods=1).abs(), 
               upper_shadow(btc), lower_shadow(btc)], axis=1)
y_btc = btc.Target

X_eth = pd.concat([log_return(eth.VWAP,periods=5), log_return(eth.VWAP,periods=1).abs(), 
               upper_shadow(eth), lower_shadow(eth)], axis=1)
y_eth = eth.Target

train_window = [totimestamp("01/05/2021"), totimestamp("30/05/2021")]
test_window = [totimestamp("01/06/2021"), totimestamp("30/06/2021")]

# Dividindo os dados em para teste e treino, variaveis X e y 
# we aim to build simple regression models using a window_size of 1
X_btc_train = X_btc.loc[train_window[0]:train_window[1]].fillna(0).to_numpy()  # filling NaN's with zeros
y_btc_train = y_btc.loc[train_window[0]:train_window[1]].fillna(0).to_numpy()  

X_btc_test = X_btc.loc[test_window[0]:test_window[1]].fillna(0).to_numpy() 
y_btc_test = y_btc.loc[test_window[0]:test_window[1]].fillna(0).to_numpy() 

X_eth_train = X_eth.loc[train_window[0]:train_window[1]].fillna(0).to_numpy()  
y_eth_train = y_eth.loc[train_window[0]:train_window[1]].fillna(0).to_numpy()  

X_eth_test = X_eth.loc[test_window[0]:test_window[1]].fillna(0).to_numpy() 
y_eth_test = y_eth.loc[test_window[0]:test_window[1]].fillna(0).to_numpy() 


# Standarlization (Processo para colocar diferentes variaveis na mesma escala)

# Simple preprocessing of the data 
scaler = StandardScaler()

X_btc_train_scaled = scaler.fit_transform(X_btc_train)
X_btc_test_scaled = scaler.transform(X_btc_test)

X_eth_train_scaled = scaler.fit_transform(X_eth_train)
X_eth_test_scaled = scaler.transform(X_eth_test)

#Regressão Linear

# Implementando ML basicoo, um por ativo
lr = LinearRegression()
lr.fit(X_btc_train_scaled,y_btc_train)
y_pred_lr_btc = lr.predict(X_btc_test_scaled)

lr.fit(X_eth_train_scaled,y_eth_train)
y_pred_lr_eth = lr.predict(X_eth_test_scaled)

# Implementando base complexa (multiple output regression model)


# Concatenando os ativos
X_both_train = np.concatenate((X_btc_train_scaled, X_eth_train_scaled), axis=1)
X_both_test = np.concatenate((X_btc_test_scaled, X_eth_test_scaled), axis=1)
y_both_train = np.column_stack((y_btc_train, y_eth_train))
y_both_test = np.column_stack((y_btc_test, y_eth_test))

# Definindo o Multi-Output e preenchendo com dados
mlr = MultiOutputRegressor(LinearRegression())
lr.fit(X_both_train,y_both_train)
y_pred_lr_both = lr.predict(X_both_test)

print('Test score da base em Regressão Linear: BTC', f"{np.corrcoef(y_pred_lr_btc, y_btc_test)[0,1]:.2f}", 
                                ', ETH', f"{np.corrcoef(y_pred_lr_eth, y_eth_test)[0,1]:.2f}")
print('Test score para multiplos outputs da base em Regressão Linear: BTC', f"{np.corrcoef(y_pred_lr_both[:,0], y_btc_test)[0,1]:.2f}", 
                                                ', ETH', f"{np.corrcoef(y_pred_lr_both[:,1], y_eth_test)[0,1]:.2f}")


