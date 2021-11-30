import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import array

data = pd.read_csv('https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv')
print(data.head())

temp = data.reset_index()['Close']
print(plt.plot(temp))

temp = data.reset_index()['Close']
plt.plot(temp)

print(data.head())
print(temp.shape)

scaler = MinMaxScaler(feature_range=(0,1))
temp = scaler.fit_transform(np.array(temp).reshape(-1,1))

print(temp.shape)

# Doing train test split (70-30) sequencially 
data = data.sort_values(by="Date")
ntrain = int(len(temp)*0.7)
train, test = temp[0:ntrain,:], temp[ntrain:len(temp),:1]

# Convert array of values to dataset matrix
def dataset(df,time_step=1):
    dx,dy = [],[]
    for i in range(len(df)-time_step-1):
        dx.append(df[i:(i+time_step),0])
        dy.append(df[i+time_step,0])
    return np.array(dx),np.array(dy)


# Reshaping the Dataset in 3 parts
time_step = 50
x_train, y_train = dataset(train,time_step)
x_test, y_test = dataset(test,time_step)

# Checking the Shape
print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)

# Reshaping the data to numpy array as per requirement for LSTM
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

# Checking the Shape
print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)

model = Sequential()
model.add(LSTM(64,return_sequences = True,input_shape = (50,1)))
model.add(LSTM(32))
model.add(Dense(16,activation='relu'))
model.add(Dense(1))

model.summary()
model.compile(loss='mse',optimizer='adam')

earlystopping = callbacks.EarlyStopping(monitor ="val_loss",mode ="min",patience = 3,restore_best_weights = True)
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs = 15,batch_size = 32,verbose=1,callbacks =[earlystopping])

#Prediction on test Data
yp_train = model.predict(x_train)
yp_test = model.predict(x_test)

#Inverse transformation
yp_train = scaler.inverse_transform(yp_train)
yp_test = scaler.inverse_transform(yp_test)

# MAE
print("MAE for Train Data: {}".format(mean_absolute_error(y_train,yp_train)))
print("MAE for Test Data: {}\n".format(mean_absolute_error(y_test,yp_test)))

# MSE
print("MSE for Train Data: {}".format(mean_squared_error(y_train,yp_train)))
print("MSE for Test Data: {}\n".format(mean_squared_error(y_test,yp_test)))


#Plotando a previsão do Output

# Shifting train predictions for plotting
look_back=50
yp_train_plot = np.empty_like(temp)
yp_train_plot[:,:] = np.nan
yp_train_plot[look_back:len(yp_train)+look_back,:] = yp_train

# Shifting test predictions for plotting
yp_test_plot = np.empty_like(temp)
yp_test_plot[:, :] = np.nan
yp_test_plot[len(yp_train)+(look_back*2)+1:len(temp)-1,:] = yp_test

# Plotting predictions
plt.plot(scaler.inverse_transform(temp))
plt.plot(yp_train_plot)
plt.plot(yp_test_plot)
plt.show()

#Prevendo e estimando para os proximos 30 dias

x_input = test[(len(test)-50):].reshape(1,-1) # We take last 50 days data from test data for our future prediction 
temp_input = list(x_input)
temp_input = temp_input[0].tolist() # Test data

# Predictions for next 30 days

lstm_op = []
n_steps = 50
i = 0
while(i < 30):
    if(len(temp_input)>50):
        x_input=np.array(temp_input[1:])
        
        #print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yp = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yp))
        
        temp_input.extend(yp[0].tolist())
        temp_input=temp_input[1:]
        lstm_op.extend(yp.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1,n_steps,1))
        yp = model.predict(x_input,verbose=0)
        
        temp_input.extend(yp[0].tolist())
        lstm_op.extend(yp.tolist())
        i=i+1

day_new = np.arange(1,51)
day_pred=np.arange(51,51+30)

plt.plot(day_new,scaler.inverse_transform(temp[len(temp)-50:]))
plt.plot(day_pred,scaler.inverse_transform(lstm_op))
