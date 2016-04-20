from scikits.audiolab import Sndfile
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import pandas as pd
import  numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
import os
import glob
from enum import Enum


os.chdir(os.path.expanduser('/home/mfcc/Desktop/data_mining_music_classification/genres/'))
no_of_audio = len(glob.glob('*/*.au'))
Genre = Enum('Genre','classical jazz pop rock')
data = [None]*no_of_audio
data_with_frames = {}
labels = [None]*no_of_audio
file_index = {}
index = 0;


hop_size = 0.01
frame_size = 0.025

for dir in glob.glob("*"):
    if(os.path.isdir(dir) and dir in Genre.__members__):
        for file in glob.glob(dir+"/*.au"):
            f = Sndfile(file,'r')
            Fs = f.samplerate
            audio = f.read_frames(f.nframes)
            label = Genre[dir].value
            features = audioFeatureExtraction.stFeatureExtraction(audio,Fs,Fs*frame_size,Fs*frame_size)
            no_of_features = features.shape[1]
            mean_of_features =np.mean(features,axis=1)
            features_with_label = np.concatenate((mean_of_features,[label]),axis=0)
            new_row = pd.DataFrame([features_with_label],columns=['ZCR']+['EnergyEntropy']+['SpectralEntropy']+['Flux']+['MFCC']*13+['Label'])
            data_with_frames[file] = new_row
            file_index[file] = index
            data[index] = mean_of_features; labels[index] = label
	    if(index==0):
		print
		print
		print 'starting feature extraction'
	    if(index==25):
		print 'yes, the extraction is a bit slow, plz improve it'
	    if(index%50==0 and index!=0):
		print 'extracted features for {0} tracks'.format(index)
            index=index+1


train_data  = [];
train_data_label = [];
data_panel = pd.Panel.from_dict(data_with_frames)
for i in range(4):
    train_data = train_data+data[i*100:i*100+50]
    train_data_label = train_data_label+labels[i*100:i*100+50]

clf = svm.SVC(C=4.5,kernel='linear',decision_function_shape='ovo')
clf.fit(train_data,train_data_label)
test_data = []
test_data_label = []
for i in range(4):
    test_data = test_data+data[i*100+50:i*100+75]
    test_data_label = test_data_label+labels[i*100+50:i*100+75]
    
predict_label=clf.predict(test_data)
result = confusion_matrix(test_data_label,predict_label)
print 'confusion matrix: '
result = dict(zip(Genre.__members__,result))
print result




