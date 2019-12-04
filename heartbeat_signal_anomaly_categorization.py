import numpy as np
import pandas as pd
import os,sys
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense,Activation
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.utils.generic_utils import get_custom_objects
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

def create_dataset(X,Y):
    broj_primera=X.shape[0]
    broj_odlika=X.shape[1]
    vremenska_duzina=50
    x_oblik=(broj_primera-vremenska_duzina,vremenska_duzina+1,broj_odlika)
    y_oblik=(broj_primera-vremenska_duzina,5)
    x = np.zeros(shape=x_oblik, dtype=np.float16)
    y = np.zeros(shape=y_oblik, dtype=np.float16)
    for i in range(50,broj_primera):
        x[i-50]=X[i-50:i+1,::]
        y[i-50]=Y[i,::]
    return x,y

def kreiranje_izlaza(signali):
    broj_vrsta=signali.shape[0]
    broj_kolona=5
    oblik=(broj_vrsta,broj_kolona)
    y=np.zeros(shape=oblik)
    for i in range(broj_vrsta):
        if(signali[i,3]==0):
            y[i,0]=1
        elif(signali[i,3]==1):
            y[i,1]=1
        elif(signali[i,3]==2):
            y[i,2]=1
        elif(signali[i,3]==3):
            y[i,3]=1
        elif(signali[i,3]==4):
            y[i,4]=1
    return y

path=r"C:\Users\Computer\Desktop\master_rad\mitbih_database"
dirs=os.listdir(path)
k=0
ukupan_broj_signala=0
signali_ukupno=[]
for file1 in glob.glob(path+'/*.csv'):
      file2 = glob.glob(path + '/*.txt')
      anotacija=file2[k]
      print(k)
      print(file1)
      print(anotacija)
      signal = pd.read_csv(file1, delimiter=',', header=None)
      signal = np.array(signal)
      signal = signal[1:,::]
      anotacija= pd.read_fwf(anotacija)
      anotacija = np.array(anotacija)
      oblik = (len(anotacija), 5)
      forma = np.zeros(shape=oblik)
      for i in range(len(anotacija)):
          odbirak = anotacija[i, 1]
          tip_signala = anotacija[i, 2]
          if tip_signala == 'N':
              tip_signala_2 = 0
              anomalija = 0
          elif tip_signala == 'A':
              tip_signala_2 = 1
              anomalija = 1
          elif tip_signala == '/':
              tip_signala_2 = 2
              anomalija = 1
          elif tip_signala == 'V':
              tip_signala_2 = 3
              anomalija = 1
          elif tip_signala == 'R':
              tip_signala_2 = 4
              anomalija = 1
          else:
              tip_signala_2 = -1
              anomalija = -1
          vrednosti_dva_EKG_kanala = signal[odbirak, :]
          forma[i, 0] = odbirak
          forma[i, 1] = vrednosti_dva_EKG_kanala[1]
          forma[i, 2] = vrednosti_dva_EKG_kanala[2]
          forma[i, 3] = tip_signala_2
          forma[i, 4] = anomalija
      k=k+1
      ukupan_broj_signala=ukupan_broj_signala+forma.shape[0]
      signali_ukupno.append(forma)
signali_ukupno=np.array(signali_ukupno)
oblik=(ukupan_broj_signala,5)
signali_svi=np.zeros(shape=oblik)
brojenje=0
for i in range(signali_ukupno.shape[0]):
    podniz=signali_ukupno[i]
    for j in range(podniz.shape[0]):
        signali_svi[brojenje,:]=podniz[j,:]
        brojenje=brojenje+1
signali=[]
i=0
for i in range(signali_svi.shape[0]):
    if(signali_svi[i,4]!=-1):
        signali.append(signali_svi[i,:])
signali=np.array(signali)
print(signali.shape[0])
X=signali[::,1:4]
sc1=MinMaxScaler(feature_range=(0,1))
sc2=MinMaxScaler(feature_range=(0,1))
X[::,0]=np.reshape(sc1.fit_transform(np.reshape(X[::,0],(X.shape[0],1))),X.shape[0])
X[::,1]=np.reshape(sc2.fit_transform(np.reshape(X[::,1],(X.shape[0],1))),X.shape[0])
y=kreiranje_izlaza(signali)
print(y)
X,y=create_dataset(X,y)
print(X)
print(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8)
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],X_train.shape[2]))
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],X_test.shape[2]))
model=Sequential()
model.add(LSTM(units=30,activation='tanh',recurrent_activation='hard_sigmoid',recurrent_initializer='glorot_uniform',input_shape=(X_train.shape[1],X_train.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=45,activation='tanh',recurrent_activation='hard_sigmoid',recurrent_initializer='glorot_uniform',return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=25,activation='tanh',recurrent_activation='hard_sigmoid',recurrent_initializer='glorot_uniform'))
model.add(Dropout(0.2))
model.add(Dense(units=5,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc',f1])
path_checkpoint='najbolje_tezine_za_EKG_kategorizacija.keras'
callback_checkpoint=ModelCheckpoint(filepath=path_checkpoint,monitor='val_loss',verbose=1,save_weights_only=True,save_best_only=True)
callback_early_stopping=EarlyStopping(monitor='val_loss',patience=10,verbose=1)
callback_tensorboard=TensorBoard(log_dir='./posmatranje_logs/',histogram_freq=0,write_graph=True)
callback_reduce_lr=ReduceLROnPlateau(monitor='val_loss',factor=0.9,min_lr=1e-4,patience=5,verbose=1)
callbacks=[callback_early_stopping,callback_checkpoint,callback_tensorboard,callback_reduce_lr]
history=model.fit(X_train, y_train,epochs=100,batch_size=32,callbacks=callbacks,validation_data=(X_test,y_test),verbose=1) #broj epoha je 50, batch size je 64 bila
try:
    model.load_weights(path_checkpoint)
    print("Successfully loaded the best weights")
except Exception as error:
    print("Error")
    print(error)
mse, acc, F1 = model.evaluate(X_test, y_test)
prediction=model.predict(X_test)
y_pred=np.zeros(shape=(prediction.shape[0],prediction.shape[1]))
for i in range(prediction.shape[0]):
    max_indeks=np.argmax(prediction[i,::])
    y_pred[i,max_indeks]=1
print('mean_squared_error :', mse)
print('accuracy:', acc)
print('F1:', F1)
history.history.keys()
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['f1'])
plt.plot(history.history['val_f1'])
plt.legend(labels=['loss','val_loss','f1','val_f1'],loc='best')
cm=confusion_matrix(np.argmax(y_test,axis=1),np.argmax(y_pred,axis=1))
print(cm)
