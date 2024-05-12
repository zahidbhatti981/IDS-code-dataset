# %% [markdown]
# # **Imports**

# %%
# importing required libraries
import numpy as np
import pandas as pd
import pickle # saving and loading trained model
from os import path

# importing required libraries for normalizing data
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# importing library for plotting
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from keras.layers import Dense, LSTM, MaxPool1D, Flatten, Dropout # importing dense layer
from keras.models import Sequential #importing Sequential layer
from keras.layers import Input
from keras.models import Model
# representation of model layers
from keras.utils.vis_utils import plot_model

# %% [markdown]
# ## Reading Data

# %%
feature=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
          "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells",
          "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
          "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count", 
          "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
          "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]

# %%
train='../input/nslkdd/KDDTrain+.txt'
test='../input/nslkdd/KDDTest+.txt'
test21='../input/nslkdd/KDDTest-21.txt'
train_data=pd.read_csv(train,names=feature)
test_data=pd.read_csv(test,names=feature)
test_data21 = pd.read_csv(test21, names= feature)
data= pd.concat([train_data, test_data], ignore_index=True)

# %%
data

# %%
# remove attribute 'difficulty_level'
data.drop(['difficulty'],axis=1,inplace=True)

# %% [markdown]
# ## Exploring Data

# %%
data.info()

# %%
import pandas as pd

# Assuming 'data' is your DataFrame with the column named 'label'
unique_labels = data['label'].unique()

print("Unique values in the 'label' column:")
for label in unique_labels:
    print(label)


# %%
data.describe().T

# %%
# number of attack labels 
data['label'].value_counts()

# %%
# Redistribute across common attack class
def change_label(df):
  df.label.replace(['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm'],'Dos',inplace=True)
  df.label.replace(['ftp_write','guess_passwd','httptunnel','imap','multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xlock','xsnoop'],'R2L',inplace=True)      
  df.label.replace(['ipsweep','mscan','nmap','portsweep','saint','satan'],'Probe',inplace=True)
  df.label.replace(['buffer_overflow','loadmodule','perl','ps','rootkit','sqlattack','xterm'],'U2R',inplace=True)

# %%
change_label(data)

# %%
# distribution of attack classes
data.label.value_counts()

# %% [markdown]
# ### Protocol

# %%
# creating a dataframe with multi-class labels (Dos,Probe,R2L,U2R,normal)
label = pd.DataFrame(data.label)

# %%
# using standard scaler for normalizing
std_scaler = StandardScaler()
def standardization(df,col):
    for i in col:
        arr = df[i]
        arr = np.array(arr)
        df[i] = std_scaler.fit_transform(arr.reshape(len(arr),1))
    return df

numeric_col = data.select_dtypes(include='number').columns
data = standardization(data,numeric_col)


# %%
# label encoding (0,1,2,3,4) multi-class labels (Dos,normal,Probe,R2L,U2R)
le2 = preprocessing.LabelEncoder()
enc_label = label.apply(le2.fit_transform)
data['intrusion'] = enc_label
print(data.shape)
data

# %%
data.drop(labels= ['label'], axis=1, inplace=True)
print(data.shape)


# %%
# one-hot-encoding categorical columns
data = pd.get_dummies(data,columns=['protocol_type','service','flag'],prefix="",prefix_sep="")  
print(data.shape)

# %%
y_data= data['intrusion']
X_data= data.drop(labels=['intrusion'], axis=1)

print('X_train has shape:',X_data.shape,'\ny_train has shape:',y_data.shape)

# %%
from sklearn.preprocessing import LabelBinarizer
y_data = LabelBinarizer().fit_transform(y_data)

X_data=np.array(X_data)
y_data=np.array(y_data)

# %%
X_train, X_test, y_train, y_test = train_test_split(X_data,y_data, test_size=0.20, random_state=42)
print(X_train.shape,'\n',X_test.shape)

# %%
# reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, ( X_train.shape[0], 1 , X_train.shape[1] ))
X_test = np.reshape(X_test, ( X_test.shape[0], 1,  X_test.shape[1] ))

# %%
# model = Sequential() # initializing model
# model.add(LSTM(64,return_sequences=True,input_shape = (1, X_train.shape[2])))
# model.add(Dropout(0.2))
# model.add(LSTM(64,return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(64,return_sequences=True))
# model.add(Flatten())
# model.add(Dense(units=50))
# # output layer with softmax activation
# model.add(Dense(units=5,activation='softmax'))

# %%
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Flatten, Dense

# Initialize the model
model = Sequential()

# Add the first LSTM layer
model.add(LSTM(64, return_sequences=True, input_shape=(1, X_train.shape[2])))
model.add(Dropout(0.2))

# Add the second LSTM layer
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))

# Add the third LSTM layer
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))

# Flatten the output from LSTM layers
model.add(Flatten())

# Add a Dense layer
model.add(Dense(units=50, activation='relu'))

# Add the output layer with softmax activation
model.add(Dense(units=5, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the summary of the model
model.summary()


# %%
# defining loss function, optimizer, metrics and then compiling model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# %%
# summary of model layers
model.summary()

# %%
# training the model on training dataset
history = model.fit(X_train, y_train, epochs=100, batch_size=5000,validation_split=0.2)

# %%
# predicting target attribute on testing dataset
test_results = model.evaluate(X_test, y_test, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]*100}%')

# %%
# Plot of accuracy vs epoch for train and test dataset
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Plot of accuracy vs epoch for train and test dataset")
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

# %%
# Plot of loss vs epoch for train and test dataset
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Plot of loss vs epoch for train and test dataset")
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# %%
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix,multilabel_confusion_matrix

# Predictions
y_pred = model.predict(X_test)
y_pred_classes = np.round(y_pred)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_classes))

# Print confusion matrix


# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix

# Assuming y_test and y_pred_classes are your true labels and predicted labels respectively
# Calculate confusion matrix
conf_mat = multilabel_confusion_matrix(y_test, y_pred_classes)

# Flatten the 3D array into a 2D array
flat_conf_mat = conf_mat.reshape(-1, conf_mat.shape[2])

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(flat_conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


# %%
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix

# Assuming y_test and y_pred_classes are your true labels and predicted labels respectively
# Calculate confusion matrix
conf_mat = multilabel_confusion_matrix(y_test, y_pred_classes)

# Visualize confusion matrices
plt.figure(figsize=(15, 10))

for i in range(len(conf_mat)):
    plt.subplot(2, 3, i + 1)
    sns.heatmap(conf_mat[i], annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted 0', 'Predicted 1'], 
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title(f'Confusion Matrix - Class {i}')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')

plt.tight_layout()
plt.show()



