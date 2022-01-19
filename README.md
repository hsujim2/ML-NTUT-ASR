# ML-NTUT-ASR
It's a homework in NTUT, for Taiwanese Speech Recognition

# Environment
>cpu:i5-12400<br>
>gpu:rtx 3070<br>
>system:ubuntu 21.10 LTS<br>
>python:3.9.7<br>

other software's version are in requirement.txt file

# Preparation
    sudo apt update && sudo apt upgrade
    sudo apt install python3 git vim python3-pip
    source path-to-venv/bin/activate
    pip3 install -r requirement.txt
    kaggle competitions download -c machine-learningntut-2021-autumn-asr
    unzip machine-learningntut-2021-autumn-asr.zip
    rm -r ML@NTUT-2021-Autumn-ASR/train/PexHeader
done!<br>
# Execute python code
    python3 train.py
model and dictionary file will be save, then run test data<br>

    python3 test.py
will generate or update speech_predict.csv file<br>
# Results
training<br>
![](https://i.imgur.com/RdhHjqI.png)
![](https://i.imgur.com/suALMZA.png)
testing<br>
![](https://i.imgur.com/2rAIboK.png)
kaggle result
![](https://i.imgur.com/lRW1X8N.png)
# Code
import<br>
```Python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Conv1D, Lambda, Add, Multiply, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import csv
import random
import pickle
import glob
from tqdm import tqdm
import os
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import librosa.display
from IPython.display import Audio
```
setting up some parameters
```python
batch_size = 4#set too large will face OOM problem
epochs = 30
mecc_dim = 15#features length return from python_speech_features
mfcc_dim = 15#using for trainning, need to be same as mecc_dim
train_split = 0.9#choose how much data to be split to train, others to validation
num_blocks = 4#numbers of cnn block
filters = 64#numbers of dense layers
```
some class and functions
```python
########################save best result########################3
class SaveBestModel(tf.keras.callbacks.Callback):
    def __init__(self, save_best_metric='val_loss', this_max=False):
        self.save_best_metric = save_best_metric
        self.max = this_max
        if this_max:
            self.best = float('-inf')
        else:
            self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        metric_value = logs[self.save_best_metric]
        if self.max:
            if metric_value > self.best:
                self.best = metric_value
                self.best_weights = self.model.get_weights()

        else:
            if metric_value < self.best:
                self.best = metric_value
                self.best_weights= self.model.get_weights()

##############################some functions######################
def get_wave(wav_path):#get wave files using per-found path
    wav_files = []
    for (dirpath, dirnames, filenames) in os.walk(wav_path):
        for filename in filenames:
            if filename.endswith('.wav') or filename.endswith('.WAV'):
                filename_path = os.path.join(dirpath, filename)
                wav_files.append(filename_path)
    return wav_files

def get_tran_texts(wav_files, train_path):
    tran_texts = []
    for wav_file in wav_files:#search all wave files in extension .wav
        basename = os.path.basename(wav_file)
        x = os.path.splitext(basename)[0]
        tran_file = os.path.join(train_path,x+'.txt')
        if os.path.exists(tran_file) is False:
            return None
        fd = open(tran_file, 'r')#if exists, read it!
        text = fd.readline()
        tran_texts.append(text.split('\n')[0])
        fd.close()
    return tran_texts

def load_and_trim(path):#do some pre-training operation
    audio, sr = librosa. load( path)
    energy = librosa.feature.rms(audio)
    frames = np.nonzero(energy >= np. max(energy) / 5)
    indices = librosa.core.frames_to_samples(frames)[ 1]
    audio = audio[indices[ 0]:indices[ -1]] if indices.size else audio[ 0: 0]
    return audio, sr

#loss function
def calc_ctc_loss(args):
    y, yp, ypl, yl = args
    return K.ctc_batch_cost(y, yp, ypl, yl)

#set batch size
def batch_generator(x, y, batch_size=batch_size):
    offset = 0
    while True:
        offset += batch_size
        if offset == batch_size or offset >= len(x):
            data_index = np.arange(len(x))
            np.random.shuffle(data_index)
            x = [x[i] for i in data_index]
            y = [y[i] for i in data_index]
            offset = batch_size

        X_data = x[offset - batch_size: offset]
        Y_data = y[offset - batch_size: offset]

        X_maxlen = max([X_data[i].shape[0] for i in range(batch_size)])
        Y_maxlen = max([len(Y_data[i]) for i in range(batch_size)])

        X_batch = np.zeros([batch_size, X_maxlen, mfcc_dim])
        Y_batch = np.ones([batch_size, Y_maxlen]) * len(char2id)
        X_length = np.zeros([batch_size, 1], dtype='int32')
        Y_length = np.zeros([batch_size, 1], dtype='int32')

        for i in range(batch_size):
            X_length[i, 0] = X_data[i].shape[0]
            X_batch[i, :X_length[i, 0], :] = X_data[i]

            Y_length[i, 0] = len(Y_data[i])
            Y_batch[i, :Y_length[i, 0]] = [char2id[c] for c in Y_data[i]]

        inputs = {'X': X_batch, 'Y': Y_batch, 'X_length': X_length, 'Y_length': Y_length}
        outputs = {'ctc': np.zeros([batch_size])}

        yield (inputs, outputs)

#cnn layer
def conv1d(inputs, filters, kernel_size, dilation_rate):
    return Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='causal', activation=None,
                  dilation_rate=dilation_rate)(inputs)

#Normalization
def batchnorm(inputs):
    return BatchNormalization()(inputs)

#Activation layer
def activation(inputs, activation):
    return Activation(activation)(inputs)

#dense layer
def res_block(inputs, filters, kernel_size, dilation_rate):
    hf = activation(batchnorm(conv1d(inputs, filters, kernel_size, dilation_rate)), 'tanh')
    hg = activation(batchnorm(conv1d(inputs, filters, kernel_size, dilation_rate)), 'sigmoid')
    h0 = Multiply()([hf, hg])

    ha = activation(batchnorm(conv1d(h0, filters, 1, 1)), 'tanh')
    hs = activation(batchnorm(conv1d(h0, filters, 1, 1)), 'tanh')

    return Add()([ha, inputs]), hs
#To prevent keras from using all vram
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```
reading files and pre-processing
```python
wav_files = get_wave('/home/hsujim/ML@NTUT-2021-Autumn-ASR/train')
tran_text = get_tran_texts(wav_files,'/home/hsujim/ML@NTUT-2021-Autumn-ASR/train/txt')
#########################sound pre-processing#######################
features = []
for i in tqdm(range(len(wav_files))):#read wave and pre-processing
    path = wav_files[i]
    #print("read:" + wav_files[i])
    audio, sr = load_and_trim(path)
    features.append(mfcc(audio, numcep=mecc_dim, samplerate=sr,winlen=0.025,winstep=0.01,
    nfilt=26,nfft=551,lowfreq=0,highfreq=None,preemph=0.79,ceplifter=22,appendEnergy=True))
#print(len(features), features[0].shape)
samples = random.sample(features, 100)
samples = np.vstack(samples)

mfcc_mean = np.mean(samples, axis=0)
mfcc_std = np.std(samples, axis=0)
print(mfcc_mean)
print(mfcc_std)

features = [(feature - mfcc_mean) / (mfcc_std + 1e-14) for feature in features]
#-min/std, a kind of normalization

#########################text pre-processing#######################
chars = {}
for text in tran_text:
    for c in text:
        chars[c] = chars.get(c, 0) + 1

chars = sorted(chars.items(), key=lambda x: x[1], reverse=True)
chars = [char[0] for char in chars]
print(len(chars), chars[:100])

char2id = {c: i for i, c in enumerate(chars)}
id2char = {i: c for i, c in enumerate(chars)}
data_index = np.arange(len(wav_files))
np.random.shuffle(data_index)

######################preparing datas for trainning##################3
train_size = int(train_split * len(wav_files))
test_size = len(wav_files) - train_size
train_index = data_index[:train_size]
test_index = data_index[train_size:]
#here read preprocessing data from features
X_train = [features[i] for i in train_index]
Y_train = [tran_text[i] for i in train_index]
X_test = [features[i] for i in test_index]
Y_test = [tran_text[i] for i in test_index]
```
setting up model and start training
```python
#####################setting up model###########################
X = Input(shape=(None, mfcc_dim,), dtype='float32', name='X')
Y = Input(shape=(None,), dtype='float32', name='Y')
X_length = Input(shape=(1,), dtype='int32', name='X_length')
Y_length = Input(shape=(1,), dtype='int32', name='Y_length')
h0 = activation(batchnorm(conv1d(X, filters, 1, 1)), 'tanh')
shortcut = []
for i in range(num_blocks):
    for r in [1, 2, 4, 8, 16]:
        h0, s = res_block(h0, filters, 14, r)
        shortcut.append(s)

h1 = activation(Add()(shortcut), 'relu')
h1 = activation(batchnorm(conv1d(h1, filters, 1, 1)), 'relu')
Y_pred = activation(batchnorm(conv1d(h1, len(char2id) + 1, 1, 1)), 'softmax')#out layer
sub_model = Model(inputs=X, outputs=Y_pred)
ctc_loss = Lambda(calc_ctc_loss, output_shape=(1,), name='ctc')([Y, Y_pred, X_length, Y_length])
model = Model(inputs=[X, Y, X_length, Y_length], outputs=ctc_loss)
optimizer = SGD(lr=0.02, momentum=0.9, nesterov=True, clipnorm=5)
model.compile(loss={'ctc': lambda ctc_true, ctc_pred: ctc_pred}, optimizer=optimizer)#compile
save_best_model = SaveBestModel()
checkpointer = ModelCheckpoint(filepath='asr.h5', verbose=0)
lr_decay = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=1, min_lr=0.000)

#######################starting trainning#############################
history = model.fit_generator(
    generator=batch_generator(X_train, Y_train),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs,
    validation_data=batch_generator(X_test, Y_test),
    validation_steps=len(X_test) // batch_size,
    callbacks=[save_best_model])
```
saving and display
```python
#########################saving and display##########################
model.set_weights(save_best_model.best_weights)#restore to the best one
sub_model.save('asr.h5')#save models
with open('dictionary.pkl', 'wb') as fw:#save dictionary
    pickle.dump([char2id, id2char, mfcc_mean, mfcc_std], fw)
#plot results
train_loss = history.history['loss']
valid_loss = history.history['val_loss']
plt.plot(np.linspace(1, epochs, epochs), train_loss, label='train')
plt.plot(np.linspace(1, epochs, epochs), valid_loss, label='valid')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
```
test
```python
folder_path = '/home/hsujim/ML@NTUT-2021-Autumn-ASR/test-shuf/'
mfcc_dim = 15
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
```
