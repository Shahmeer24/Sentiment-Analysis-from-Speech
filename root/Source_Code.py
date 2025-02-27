import librosa #used for audio processing tasks
import librosa.display #provides functions for visualizing audio data
import numpy as np #used for numerical computations
import matplotlib.pyplot as plt #used for creating static and interactive visualizations
from matplotlib.pyplot import specgram
import pandas as pd #package for data manipulation and library analysis
import glob #used for file operations and pathname pattern matching
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #used for normalization of data
from sklearn.neighbors import KNeighborsClassifier #required for KNN model
from sklearn.svm import SVC #required for SVM model
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.externals import joblib #used for model serialization
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import IPython.display as ipd #used for displaying audio signals
from tqdm import tqdm #progress bar
import seaborn as sns #used for visualization
import pickle #used for serialization and of Python objects
import os #used for file operations and manipulating environment variables
import sys

import warnings
# ignore warnings 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Traversing the directory tree rooted at the input directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)

# Defining the paths to the datasets
TESS = "/kaggle/input/toronto-emotional-speech-set-tess/tess toronto emotional speech set data/TESS Toronto emotional speech set data/"
RAV = "/kaggle/input/ravdess-emotional-speech-audio/audio_speech_actors_01-24/"
SAVEE = "/kaggle/input/surrey-audiovisual-expressed-emotion-savee/ALL/"
CREMA = "/kaggle/input/cremad/AudioWAV/"

#Exploring SAVEE Dataset
dir_list = os.listdir(SAVEE)

# List initialization to store emotions and file paths
emotion =[]
path = []

for i in dir_list:
    if i[-8:-6]=='_a':
        emotion.append('male_angry')
    elif i[-8:-6]=='_d':
        emotion.append('male_disgust')
    elif i[-8:-6]=='_f':
        emotion.append('male_fear')
    elif i[-8:-6]=='_h':
        emotion.append('male_happy')
    elif i[-8:-6]=='_n':
        emotion.append('male_neutral')
    elif i[-8:-6]=='sa':
        emotion.append('male_sad')
    elif i[-8:-6]=='su':
        emotion.append('male_surprise')
    else:
        emotion.append('male_error') 
    path.append(SAVEE + i)

# Dataframe creation to store emotion labels, file paths and source dataset
SAVEE_df = pd.DataFrame(emotion, columns = ['labels'])
SAVEE_df['source'] = 'SAVEE'
SAVEE_df = pd.concat([SAVEE_df, pd.DataFrame(path, columns = ['path'])], axis = 1)
print(SAVEE_df.labels.value_counts())

#Exploring RAVDESS Dataset
dir_list = os.listdir(RAV)
dir_list.sort()

# List initialization to store emotions and file paths
emotion = []
gender = []
path = []
for i in dir_list:
    fname = os.listdir(RAV + i)
    for f in fname:
        part = f.split('.')[0].split('-')
        emotion.append(int(part[2]))
        temp = int(part[6])
        if temp%2 == 0:
            temp = "female"
        else:
            temp = "male"
        gender.append(temp)
        path.append(RAV + i + '/' + f)
        
# Dataframe creation to store emotion labels, file paths and source dataset
RAV_df = pd.DataFrame(emotion)
RAV_df = RAV_df.replace({1:'neutral', 2:'neutral', 3:'happy', 4:'sad',
                         5:'angry', 6:'fear', 7:'disgust', 8:'surprise'})
RAV_df = pd.concat([pd.DataFrame(gender),RAV_df],axis=1)
RAV_df.columns = ['gender','emotion']
RAV_df['labels'] =RAV_df.gender + '_' + RAV_df.emotion
RAV_df['source'] = 'RAVDESS'  
RAV_df = pd.concat([RAV_df,pd.DataFrame(path, columns = ['path'])],axis=1)
RAV_df = RAV_df.drop(['gender', 'emotion'], axis=1)
print(RAV_df.labels.value_counts())


#Exploring TESS Dataset
dir_list = os.listdir(TESS)
dir_list.sort()

# List initialization to store emotions and file paths
path = []
emotion = []

for i in dir_list:
    fname = os.listdir(TESS + i)
    for f in fname:
        if i == 'OAF_angry' or i == 'YAF_angry':
            emotion.append('female_angry')
        elif i == 'OAF_disgust' or i == 'YAF_disgust':
            emotion.append('female_disgust')
        elif i == 'OAF_Fear' or i == 'YAF_fear':
            emotion.append('female_fear')
        elif i == 'OAF_happy' or i == 'YAF_happy':
            emotion.append('female_happy')
        elif i == 'OAF_neutral' or i == 'YAF_neutral':
            emotion.append('female_neutral')                                
        elif i == 'OAF_Pleasant_surprise' or i == 'YAF_pleasant_surprised':
            emotion.append('female_surprise')               
        elif i == 'OAF_Sad' or i == 'YAF_sad':
            emotion.append('female_sad')
        else:
            emotion.append('Unknown')
        path.append(TESS + i + "/" + f)

# Dataframe creation to store emotion labels, file paths and source dataset
TESS_df = pd.DataFrame(emotion, columns = ['labels']) 
TESS_df['source'] = 'TESS'
TESS_df = pd.concat([TESS_df,pd.DataFrame(path, columns = ['path'])],axis=1)
print(TESS_df.labels.value_counts())

#Exploring CREMA-D Dataset
dir_list = os.listdir(CREMA)
dir_list.sort()

# List initialization to store emotions and file paths
gender = []
emotion = []
path = []
female = [1002,1003,1004,1006,1007,1008,1009,1010,1012,1013,1018,1020,1021,1024,1025,1028,
          1029,1030,1037,1043,1046,1047,1049,1052, 1053,1054,1055,1056,1058,1060,1061,1063, 1072,1073,1074,1075,1076,1078,1079,1082,
          1084,1089,1091]

for i in dir_list: 
    part = i.split('_')
    if int(part[0]) in female:
        temp = 'female'
    else:
        temp = 'male'
    gender.append(temp)
    if part[2] == 'SAD' and temp == 'male':
        emotion.append('male_sad')
    elif part[2] == 'ANG' and temp == 'male':
        emotion.append('male_angry')
    elif part[2] == 'DIS' and temp == 'male':
        emotion.append('male_disgust')
    elif part[2] == 'FEA' and temp == 'male':
        emotion.append('male_fear')
    elif part[2] == 'HAP' and temp == 'male':
        emotion.append('male_happy')
    elif part[2] == 'NEU' and temp == 'male':
        emotion.append('male_neutral')
    elif part[2] == 'SAD' and temp == 'female':
        emotion.append('female_sad')
    elif part[2] == 'ANG' and temp == 'female':
        emotion.append('female_angry')
    elif part[2] == 'DIS' and temp == 'female':
        emotion.append('female_disgust')
    elif part[2] == 'FEA' and temp == 'female':
        emotion.append('female_fear')
    elif part[2] == 'HAP' and temp == 'female':
        emotion.append('female_happy')
    elif part[2] == 'NEU' and temp == 'female':
        emotion.append('female_neutral')
    else:
        emotion.append('Unknown')
    path.append(CREMA + i)


#Dataframe creation to store emotion labels, file paths and source dataset
CREMA_df = pd.DataFrame(emotion, columns = ['labels'])
CREMA_df['source'] = 'CREMA'
CREMA_df = pd.concat([CREMA_df,pd.DataFrame(path, columns = ['path'])],axis=1)
print(CREMA_df.labels.value_counts())
print() # for an extra line


# Concatenate dataframes from different emotion datasets into a single dataframe
df = pd.concat([SAVEE_df, RAV_df, TESS_df, CREMA_df], axis = 0)

# Count the occurrences of each audio label
audio_label_count = df.labels.value_counts()
print(audio_label_count)
print(df.to_csv("Data_path.csv",index=False)) # Saving the combined dataframe to a CSV file


# Generating bar plot
colors = ['blue', 'green', 'red', 'purple', 'orange',
          'yellow', 'pink', 'brown', 'gray', 'cyan',
          'magenta', 'olive', 'teal', 'navy']
pastel_colors = sns.color_palette("muted", len(colors))

plt.figure(figsize=(22, 10))
plt.bar(audio_label_count.index, audio_label_count.values, color=pastel_colors)
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Audio Label Count')
plt.xticks(audio_label_count.index)
plt.show()

# Count the occurrences of each emotion label
emotion_counts = df['labels'].value_counts()

# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=140, wedgeprops=dict(width=0.4))
# Draw a white circle at the center to make it a "donut" chart
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


#Data Processing
ref = pd.read_csv("/kaggle/working/Data_path.csv")
num_mfcc = 20  # Number of MFCC coefficients
num_files = 12162  # Number of audio files

# Dataframe initialization
columns = ['mfcc_' + str(i+1) for i in range(num_mfcc)]
df = pd.DataFrame(columns=columns, index=range(num_files))
pbar = tqdm(total=num_files) # Progress bar

# Loop through audio files and extract MFCC features
for idx, path in enumerate(ref.path):  
    #Loading the audio file from the specified path and extracting MFCC features
    X, sample_rate = librosa.load(path, res_type='kaiser_fast', duration=2.5, sr=44100, offset=0.5)
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=num_mfcc)
    
    mfccs_mean = np.mean(mfccs, axis=1)  # Compute mean of MFCC coefficients along each column
    df.loc[idx] = mfccs_mean # Storing the mean coefficients in the dataframe
    pbar.update(1)
    
pbar.close()

#Splitting and Normalizing the Dataset
extracted_features = pd.read_csv("/kaggle/working/features.csv")

X_train, X_test, y_train, y_test = train_test_split(extracted_features.drop(['path','labels','source'],
                                                                   axis=1)
                                                    , df_concat.labels
                                                    , test_size=0.20
                                                    , shuffle=True
                                                    , random_state=42 # Random seed for reproducibility
                                                   )

print(X_train.shape)
print(X_test.shape)

# Normalization of training and testing data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Model Creation and Accuracy Measurement

#KNN Model
#KNN classifier
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')  

# Training
knn.fit(X_train_scaled, y_train)

# Predict labels 
y_pred = knn.predict(X_test_scaled)

# Evaluating the classifier and calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


#SVM Model
X_train_array = np.array(X_train)
X_test_array = np.array(X_test)

# Reshape the array if necessary
X_train_flattened = X_train_array.reshape(X_train_array.shape[0], -1)
X_test_flattened = X_test_array.reshape(X_test_array.shape[0], -1)

# Initialize SVM classifier
svm_classifier = SVC(kernel='linear')

# Train the SVM classifier
svm_classifier.fit(X_train_flattened, y_train)

# Predict labels for test data
y_pred_svm = svm_classifier.predict(X_test_flattened)

# Calculate accuracy
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Accuracy:", accuracy_svm)


#LSTM Model
# Reshape input data for LSTM
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# One-hot encode the labels
label_binarizer = LabelBinarizer()
y_train_one_hot = label_binarizer.fit_transform(y_train)
y_test_one_hot = label_binarizer.transform(y_test)
num_classes = 14

# Defining the LSTM model
lstm_model = Sequential([
    LSTM(units=128, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')  # Use 'softmax' activation for multi-class classification
])

# Compiling the model
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
history = lstm_model.fit(X_train_reshaped, y_train_one_hot, epochs=100, batch_size=16, validation_data=(X_test_reshaped, y_test_one_hot))

# Evaluating the model and calculate accuracy
loss, accuracy = lstm_model.evaluate(X_test_reshaped, y_test_one_hot)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


# Predict probabilities for the test set
y_pred_probs = lstm_model.predict(X_test_reshaped)

# Convert probabilities to class labels
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test_one_hot, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=label_binarizer.classes_, yticklabels=label_binarizer.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# Plot learning and validation curves on the same graph
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['acc'], label='Training Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.title('Learning and Validation Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy')
plt.legend()
plt.grid(True)
plt.show()


# Make predictions on the test set
y_pred = lstm_model.predict_classes(X_test_reshaped)
# Convert one-hot encoded labels to integers
y_test_classes = np.argmax(y_test_one_hot, axis=1)
# Generate classification report
classification_rep = classification_report(y_test_classes, y_pred)
# Print the classification report
print("Classification Report:")
print(classification_rep)

#Testing Unknown Sample

data, sampling_rate = librosa.load('/kaggle/input/sample-data/download.wav')
plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)
ipd.Audio('/kaggle/input/sample-data/download.wav')

X, sample_rate = librosa.load('/kaggle/input/sample-data/download.wav'
                              ,res_type='kaiser_fast'
                              ,duration=2.5
                              ,sr=44100
                              ,offset=0.5
                             )

sample_rate = np.array(sample_rate)
mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=20),axis=1)
newdf = pd.DataFrame(data=mfccs).T

X_scaled = scaler.transform(newdf)
X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# Predict emotion class
predicted_class = lstm_model.predict_classes(X_reshaped)

predicted_labels = []
for index in predicted_class:
    if index == 0:
        predicted_labels.append("female_happy")
    elif index == 1:
        predicted_labels.append("female_fear")
    elif index == 2:
        predicted_labels.append("female_sad")
    elif index == 3:
        predicted_labels.append("female_disgust")
    elif index == 4:
        predicted_labels.append("female_angry")
    elif index == 5:
        predicted_labels.append("female_neutral")
    elif index == 6:
        predicted_labels.append("male_neutral")
    elif index == 7:
        predicted_labels.append("male_sad")
    elif index == 8:
        predicted_labels.append("male_disgust")
    elif index == 9:
        predicted_labels.append("male_fear")
    elif index == 10:
        predicted_labels.append("male_happy")
    elif index == 11:
        predicted_labels.append("male_angry")
    elif index == 12:
        predicted_labels.append("female_surprise")
    elif index == 13:
        predicted_labels.append("male_surprise")

print("Predicted Emotion Class: ")
print(predicted_labels)