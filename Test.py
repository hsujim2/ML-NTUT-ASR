from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import numpy as np
import librosa
from python_speech_features import mfcc
import pickle
import glob
import os.path
import os
folder_path = '/home/hsujim/ML@NTUT-2021-Autumn-ASR/test-shuf/'

##################prepare wave files for testing#################
############((((only need run for the first time)))))############
#set=4
#os.chdir(folder_path)
#
#for filename in os.listdir():
#    os.rename(filename, filename.split('.')[0].zfill(set)+'.wav')
mfcc_dim = 13
wavs=[]
########################read data########################
for data_file in sorted(os.listdir(folder_path)):
    wavs.append(data_file)
print('1:',wavs[0])

for i in range(len(wavs)):
    wavs[i]=os.path.join('/home/hsujim/ML@NTUT-2021-Autumn-ASR/test-shuf/',wavs[i])
print('2:',wavs[0])

with open('/home/hsujim/dictionary.pkl', 'rb') as fr:
    [char2id, id2char, mfcc_mean, mfcc_std] = pickle.load(fr)

model = load_model('/home/hsujim/asr.h5')#read model

print(len(wavs))
print('wavs=0', wavs[0])

#######################testing data########################
j=0
z=1
result = []
for i in range(len(wavs)):
    audio, sr = librosa.load(wavs[i])

    energy = librosa.feature.rms(audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    X_data = mfcc(audio, sr, numcep=mfcc_dim, nfft=551)
    X_data = (X_data - mfcc_mean) / (mfcc_std + 1e-14)

    pred = model.predict(np.expand_dims(X_data, axis=0))

    pred_ids = K.eval(K.ctc_decode(pred, [X_data.shape[0]], greedy=False, beam_width=10, top_paths=1)[0][0])
    pred_ids = pred_ids.flatten().tolist()

    a = ['']*(len(pred_ids))
    x=0
    for i in pred_ids:
        if (i!=-1):
            a[x] = id2char[i]
            print(id2char[i], end='')
            x=x+1
        else:
            break
    while '' in a:
        a.remove('')
    result.append(a)
#saving result
with open('/home/hsujim/speech_predict.csv','w') as f:
    f.write('id,text\n')
    for i in range(len(wavs)):
      f.write(str(i+1) + ',' + "".join(result[i]) +'\n')

