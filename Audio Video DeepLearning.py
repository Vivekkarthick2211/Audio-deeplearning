#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install librosa')


# In[2]:


import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


filename='D:\\New folder\\UrbanSound8K\\audio\\fold1\\7383-3-0-0.wav'


# In[5]:


import IPython.display as ipd
import librosa
import librosa.display


# In[6]:


plt.figure(figsize=(14,5))
librosa_audio_data,librosa_sample_rate=librosa.load(filename)
librosa.display.waveplot(librosa_audio_data,sr=librosa_sample_rate)
ipd.Audio(filename)


# In[7]:


from scipy.io import wavfile as wav
wav_sample,wav_data=wav.read(filename)


# In[8]:


wav_sample


# In[9]:


librosa_audio_data


# In[10]:


wav_data


# In[12]:


import pandas as pd
metadata=pd.read_csv('D:\\New folder\\UrbanSound8K\\metadata\\UrbanSound8K.csv')
metadata.head(10)


# In[13]:


plt.figure(figsize=(12,4))
plt.plot(wav_data)


# In[14]:


wav_data


# In[15]:


metadata['class'].value_counts()


# In[16]:


mfccs=librosa.feature.mfcc(y=librosa_audio_data,sr=librosa_sample_rate,n_mfcc=40)
mfccs


# In[17]:


import os
import numpy as np
audio_dataset_path='D:\\New folder\\UrbanSound8K\\audio'
metadata.head(10)


# In[19]:


def feature_extractor(file):
    audio,sample_rate=librosa.load(file_name,res_type='kaiser_fast')
    mfccs_features=librosa.feature.mfcc(y=audio,sr=sample_rate,n_mfcc=40)
    mfccs_scaled_features=np.mean(mfccs_features.T,axis=0)
    return  mfccs_scaled_features 


# In[23]:


import numpy as np
from tqdm import tqdm
extracted_features=[]
for index_num,row in tqdm(metadata.iterrows()):
    file_name=os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    final_class_labels=row['class']
    data=feature_extractor(file_name)
    extracted_features.append([data,final_class_labels])


# In[106]:


extracted_features_df=pd.DataFrame(extracted_features,columns=['features','class'])
extracted_features_df.tail(5)
#extracted_features_df['class'].unique()


# In[104]:


X=np.array(extracted_features_df['features'].tolist())
y=np.array(extracted_features_df['class'].tolist())


# In[105]:


X


# In[100]:


X.shape


# In[101]:


y


# In[50]:


#from sklearn.preprocessing import LabelEncoder
#label_encoder=LabelEncoder()
#y=np.array(pd.get_dummies(y))

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
y=to_categorical(label_encoder.fit_transform(y))


# In[51]:


y


# In[52]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[53]:


X_train.shape


# In[54]:


X_test.shape


# In[55]:


y_train.shape


# # Model

# In[56]:


import tensorflow as tf
print(tf.__version__)


# In[57]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten,Activation
from tensorflow.keras.optimizers import Adam
from sklearn import metrics


# In[58]:


#classes
num_labels=y.shape[1]


# In[59]:


model=Sequential()
#first  #100 means neurons
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#second  #200 means neurons
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#third #100 means neurons
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

#final
model.add(Dense(num_labels))
model.add(Activation('softmax'))


# In[51]:


model.summary()


# In[60]:


model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')


# In[62]:


from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime

num_epochs=100
num_batch_size=32

checkpointer=ModelCheckpoint(filepath='saved_models/audio_classification.hdf5',verbose=1,save_best_only=True)
start=datetime.now()

model.fit(X_train,y_train,batch_size=num_batch_size,epochs=num_epochs,validation_data=(X_test,y_test),callbacks=[checkpointer],verbose=1)

duration=datetime.now()-start
print("train time completed in time",duration)


# In[83]:


test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])


# In[78]:


#filename="D:\\New folder\\UrbanSound8K\\audio\\fold3\\6988-5-0-4.wav"
#prediction_feature=feature_extractor(filename)
#prediction_feature=prediction_feature.reshape(1,-1)
#model.predict_classes(prediction_feature)


# In[107]:


filename="D:\\New folder\\UrbanSound8K\\audio\\7975-3-0-0.wav"

#librosa_audio_data,librosa_sample_rate=librosa.load(filename)

saudio,ssample_rate=librosa.load(filename,res_type='kaiser_fast')
mfccs_features=librosa.feature.mfcc(y=saudio,sr=ssample_rate,n_mfcc=40)
mfccs_scaled_features=np.mean(mfccs_features.T,axis=0)
print(mfccs_scaled_features)

#plt.figure(figsize=(14,5))
#librosa.display.waveplot(saudio,sr=ssample_rate)
#ipd.Audio(filename)

mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
print(mfccs_scaled_features)
print(mfccs_scaled_features.shape)
predicted_label=model.predict_classes(mfccs_scaled_features)
print(predicted_label)

prediction_class=label_encoder.inverse_transform(predicted_label)
prediction_class

