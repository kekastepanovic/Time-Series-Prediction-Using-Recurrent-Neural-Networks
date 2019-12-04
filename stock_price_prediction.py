import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau
import math
from statistics import mean


def srednje_kvadratna_greska_funkcija(predictions, y_test):
    return math.sqrt(mean(np.square(predictions - y_test)))


def create_dataset(df):
    broj_primera = df.shape[0]
    broj_odlika = df.shape[1]
    vremenska_duzina = 50
    x_oblik = (broj_primera - vremenska_duzina - 6, vremenska_duzina, broj_odlika)
    y_oblik = (broj_primera - vremenska_duzina - 6, 1)
    x = np.zeros(shape=x_oblik, dtype=np.float16)
    y = np.zeros(shape=y_oblik, dtype=np.float16)
    for i in range(50, broj_primera - 6):
        x[i - 50] = df[i - 50:i, ::]
        y[i - 50] = df[i + 2, 4]  
    return x, y


def root_mean_squared_error(y_true, y_pred):
    period_zagrevanja = 5
    y_true = y_true[period_zagrevanja:, :]
    y_pred = y_pred[period_zagrevanja:, :]
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


df = pd.read_csv('GHDX.csv')
plt.plot(df['Open'].values, label='Open')
plt.plot(df['High'].values, label='High')
plt.plot(df['Low'].values, label='Low')
plt.plot(df['Close'].values, label='Close')
plt.plot(df['Adj Close'].values, label='Adj Close')
plt.xlabel('Date')
plt.ylabel('Features')
plt.title('Open, High, Low, Close and Adj Close price features')
plt.legend()
plt.show()
# Cut of volume feature
plt.plot(df['Volume'].values, label='Volume')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('Volume price feature')
plt.show()
df = df[['Open', 'High', 'Low', 'Close', 'Adj Close']].values
dataset_train = df[:int(df.shape[0] * 0.9), ::]
dataset_test = df[int(df.shape[0] * 0.9):, ::]
dataset_train = np.array(dataset_train)
dataset_test = np.array(dataset_test)
sc1 = MinMaxScaler(feature_range=(0, 1))
sc2 = MinMaxScaler(feature_range=(0, 1))
sc3 = MinMaxScaler(feature_range=(0, 1))
sc4 = MinMaxScaler(feature_range=(0, 1))
sc5 = MinMaxScaler(feature_range=(0, 1))
# sc6=MinMaxScaler(feature_range=(0,1))
dataset_train[:, 0] = np.reshape(sc1.fit_transform(np.reshape(dataset_train[:, 0], (dataset_train.shape[0], 1))),
                                 dataset_train.shape[0])
dataset_train[:, 1] = np.reshape(sc2.fit_transform(np.reshape(dataset_train[:, 1], (dataset_train.shape[0], 1))),
                                 dataset_train.shape[0])
dataset_train[:, 2] = np.reshape(sc3.fit_transform(np.reshape(dataset_train[:, 2], (dataset_train.shape[0], 1))),
                                 dataset_train.shape[0])
dataset_train[:, 3] = np.reshape(sc4.fit_transform(np.reshape(dataset_train[:, 3], (dataset_train.shape[0], 1))),
                                 dataset_train.shape[0])
dataset_train[:, 4] = np.reshape(sc5.fit_transform(np.reshape(dataset_train[:, 4], (dataset_train.shape[0], 1))),
                                 dataset_train.shape[0])
# dataset_train[:,5]=np.reshape(sc6.fit_transform(np.reshape(dataset_train[:,5],(dataset_train.shape[0],1))),dataset_train.shape[0])
dataset_test[:, 0] = np.reshape(sc1.fit_transform(np.reshape(dataset_test[:, 0], (dataset_test.shape[0], 1))),
                                dataset_test.shape[0])
dataset_test[:, 1] = np.reshape(sc2.fit_transform(np.reshape(dataset_test[:, 1], (dataset_test.shape[0], 1))),
                                dataset_test.shape[0])
dataset_test[:, 2] = np.reshape(sc3.fit_transform(np.reshape(dataset_test[:, 2], (dataset_test.shape[0], 1))),
                                dataset_test.shape[0])
dataset_test[:, 3] = np.reshape(sc4.fit_transform(np.reshape(dataset_test[:, 3], (dataset_test.shape[0], 1))),
                                dataset_test.shape[0])
dataset_test[:, 4] = np.reshape(sc5.fit_transform(np.reshape(dataset_test[:, 4], (dataset_test.shape[0], 1))),
                                dataset_test.shape[0])
# dataset_test[:,5]=np.reshape(sc6.fit_transform(np.reshape(dataset_test[:,5],(dataset_test.shape[0],1))),dataset_test.shape[0])
x_train, y_train = create_dataset(dataset_train)
x_test, y_test = create_dataset(dataset_test)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))
model = Sequential()
model.add(LSTM(units=96, activation='tanh', recurrent_activation='hard_sigmoid', recurrent_initializer='glorot_uniform',
               return_sequences=True,
               input_shape=(x_train.shape[1], x_train.shape[2])))  # ubacujemo jednu vremensku sekvencu od 50 odbiraka
model.add(Dropout(0.2))
model.add(LSTM(units=96, activation='tanh', recurrent_dropout=0.1, recurrent_activation='hard_sigmoid',
               recurrent_initializer='glorot_uniform', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=96, activation='tanh', unit_forget_bias=True, recurrent_activation='hard_sigmoid',
               recurrent_initializer='glorot_uniform'))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(loss=root_mean_squared_error, optimizer='adam')
path_checkpoint = 'najbolje_tezine_IBM.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss', verbose=1, save_weights_only=True,
                                      save_best_only=True)
callback_early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
callback_tensorboard = TensorBoard(log_dir='./posmatranje_logs/', histogram_freq=0, write_graph=True)
callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, min_lr=1e-4, patience=5, verbose=1)
callbacks = [callback_early_stopping, callback_checkpoint, callback_tensorboard, callback_reduce_lr]
model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=callbacks, validation_data=(x_test, y_test),verbose=1)  
try:
    model.load_weights(path_checkpoint)
    print("Successfully loaded the best weights")
except Exception as error:
    print("Error")
    print(error)
predictions = model.predict(x_test)
predictions = sc1.inverse_transform(predictions)
# pd.set_option('display.html.table_schema', True)
# pd.set_option('display.precision',6)
# predikcije_pandas=pd.DataFrame(predictions)
# predikcije_pandas.to_csv(r"C:\Users\Computer\Desktop\master_rad\GHDX_sutradan_predvidjene.csv",header=None,index=None, float_format='%f')
plt.plot(df[::, 0], color="red", label="real")
plt.plot(range(len(y_train) + 50, 50 + len(y_train) + len(predictions)), predictions, color="blue", label="pred")
plt.legend()
plt.xlabel('Date')
plt.ylabel('Open')
plt.title('Total and predicted open price feature')
plt.show()
y_test = sc1.inverse_transform(y_test)
prediction = np.array(predictions)
y_real = np.array(y_test)
prediction = np.reshape(prediction, (prediction.shape[0]))
y_real = np.reshape(y_real, (y_real.shape[0]))
srednje_kvadratna_greska = srednje_kvadratna_greska_funkcija(prediction, y_real)
print(srednje_kvadratna_greska)
plt.plot(y_test, color="red", label="real")
plt.plot(predictions, color="blue", label="pred")
plt.xlabel('Date')
plt.ylabel('Adjusted closing')
plt.title('Real and predicted adjusted closing price feature')
plt.legend()
plt.show()
