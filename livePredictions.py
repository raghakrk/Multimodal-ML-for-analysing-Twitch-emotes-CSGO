"""
This file can be used to try a live prediction. 
"""

import keras
import numpy as np
import librosa
import os
import csv 
global timestamp
import gc

timestamp=0

def extractdetails(filename):
    game_name = []
    starttime = []
    csv_file = open(filename,'r')
#    csv_file.readline()
    for a, b,c,d in csv.reader(csv_file, delimiter=','):
        game_name.append(a)
        starttime.append(b)
    return game_name,starttime

def convert2sec():
    h, m, s = timestamp.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

    
def convert(seconds): 
    offset=convert2sec()
    seconds+=offset
    seconds = seconds % (24 * 3600) 
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60   
    return "%d:%02d:%02d" % (hour, minutes, seconds) 

class livePredictions:
    """
    Main class of the application.
    """

    def __init__(self, path):
        """
        Init method is used to initialize the main parameters.
        """
        self.path = path
        self.file = []
        self.predictions=[]
        self.timestamp=[]
        self.prediction_class=[]

    def load_model(self):
        """
        Method to load the chosen model.
        :param path: path to your h5 model.
        :return: summary of the model with the .summary() function.
        """
        self.loaded_model = keras.models.load_model(self.path)
        return self.loaded_model.summary()

    def makepredictions(self,file):
        """
        Method to process the files and create your features.
        """
        self.file = file
        neutral=[]
        data, sampling_rate = librosa.core.load(self.file)
        print(sampling_rate)
        for i in range(0,len(data),sampling_rate):
            if i+sampling_rate<len(data):
                temp=data[i:i+sampling_rate]
            else:
                temp=data[i:-1]
            self.timestamp.append(convert(i/sampling_rate))
            mfccs = np.mean(librosa.feature.mfcc(y=temp, sr=sampling_rate, n_mfcc=40).T, axis=0)
            x = np.expand_dims(mfccs, axis=2)
            x = np.expand_dims(x, axis=0)
            prediction = self.loaded_model.predict_classes(x)
#            temp=self.loaded_model.predict(x)
            neutral.append(self.loaded_model.predict(x)[0,1])
            self.predictions.append(prediction)
            self.prediction_class.append(self.convertclasstoemotion(prediction))
        return [np.asarray(self.predictions),np.asarray(self.prediction_class),np.asarray(self.timestamp),np.asarray(neutral)]

    @staticmethod
    def convertclasstoemotion(pred):
        """
        Method to convert the predictions (int) into human readable strings.
        """
        
        label_conversion = {'0': 'neutral',
                            '1': 'calm',
                            '2': 'Excitement', ##happy
                            '3': 'sad',
                            '4': 'angry',
                            '5': 'fearful',
                            '6': 'disgust',
                            '7': 'surprised'}

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value
        return label

# Here you can replace path and file with the path of your model and of the file 
#from the RAVDESS dataset you want to use for the prediction,
# Below, I have used a neutral file: the prediction made is neutral.
def emotion_detection(pred,filename,directory):    
    predictions,classpred,time_stamp,neutral=pred.makepredictions(directory+"\\Audio\\"+filename)
    result=np.vstack((time_stamp,classpred)).T
    fname=directory+"\\Emotions\\"+filename[:-4]+'_emotion.csv'
    np.savetxt(fname, result, fmt='%s', delimiter=',', newline='\n', header='', footer='', comments='# ', encoding="utf-8")

def main(directory,fname):  
    try:   
        os.chdir(directory)
        print("Directory changed")  
    except OSError:
        print("Can't change the Current Working Directory")
    game,start_time=extractdetails(fname)
    global timestamp
    pred = livePredictions(path='Emotion_Voice_Detection_Model.h5')
    pred.load_model()
    i=0
#    for i in range(0,len(game)):
    print('extracting ',game[i])
    timestamp=start_time[i]
    emotion_detection(pred,game[i],directory)
    gc.collect()

if __name__=="__main__":
    dir_name="D:\\CSCI 599 data\\"
    fname="extract_time.csv"
    main(dir_name,fname)
