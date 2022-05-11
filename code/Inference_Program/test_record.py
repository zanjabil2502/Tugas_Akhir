import time
start_time = time.time()

import os
import sys
import numpy as np
import argparse
import warnings
import statistics
import sounddevice as sd
from scipy.io.wavfile import write
import librosa as lb
sys.path.append(os.path.abspath('script/src'))
from feature_class import features
from DSP import classify_cough
from scipy.io import wavfile
import pickle
from sklearn.preprocessing import MinMaxScaler
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
sys.path.append(os.path.abspath('script'))
from utils.models import return_model
from utils.generic_utils import load_config

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

blockPrint()
warnings.filterwarnings("ignore")


loaded_model = pickle.load(open(os.path.join('script/models', 'cough_classifier'), 'rb'))
loaded_scaler = pickle.load(open(os.path.join('script/models','cough_classification_scaler'), 'rb'))

config_path = 'script/checkpoint/config.json'
c = load_config(config_path)

audio_class = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=1024, win_length=1024, hop_length=320, n_mels=64, f_min=0, f_max=None)

model = return_model(c)
checkpoint_path = 'script/checkpoint/best_checkpoint.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.train(False)
model.zero_grad()
model.eval()

def slice_data(start, end, raw_data,  sample_rate):
    max_ind = len(raw_data) 
    start_ind = min(int(start * sample_rate), max_ind)
    end_ind = min(int(end * sample_rate), max_ind)
    return raw_data[start_ind: end_ind]

def simple_segment(audio_input,threshold):
    Scaler = MinMaxScaler(feature_range=(0,1))
    wav, fs = lb.load(audio_input, sr=None)
    wav = lb.util.normalize(wav)
    rms = lb.feature.rms(y=wav)
    times = lb.times_like(rms, sr=fs)
    rms = rms.reshape(rms.shape[1],rms.shape[0])
    rms_norm = Scaler.fit_transform(rms)
    
    index = 0
    time_cut = []
    start = []
    end = []
    for value in rms_norm:
        if value > threshold:
            time_cut.append(times[index-3])
            try:
                if rms_norm[index+1] < threshold:
                    time_cut.append(times[index+3])

                    start.append(time_cut[0])
                    end_index = len(time_cut)
                    end.append(time_cut[end_index-1])
                    time_cut = []
            except:
                time_cut.append(times[index])

                start.append(time_cut[0])
                end_index = len(time_cut)
                end.append(time_cut[end_index-1])
                time_cut = []

        index=index+1 
    return start, end, wav, fs

def predict(filename):
    wav, sr = lb.load(filename,sr=None)
    wav = lb.util.normalize(wav)
    probability = classify_cough(wav, sr, loaded_model, loaded_scaler)
    value = round(probability*100,2)
    if value < 80:
        print('='*40)
        print('FILE '+str(filename.split('/')[-1]) + ' TIDAK TERBACA SEBAGAI BATUK')
        print('Hal ini karena batuk kurang begitu natural atau masih tercampur dengan Noise')
        print('='*40)
    elif value >= 80:
        start, end, wav, fs = simple_segment(filename,0.09)
        segm_wav = []
        wav_time = []
        for num in range(len(start)):
            #print(num)
            sliced_data = slice_data(start=start[num], end=end[num], raw_data=wav, sample_rate=fs)
            duration = lb.get_duration(y=sliced_data,sr=fs)
            if duration >= 0.3 and duration <=3:
                segm_wav.append(sliced_data)
                wav_time.append(duration)
        
        max_time = max(wav_time)
        
        if max_time < 1:
            pad_wav = []
            for value in segm_wav:
                pad = lb.util.pad_center(value,1*fs)
                pad_wav.append(pad)
        elif max_time >= 1:
            pad_wav = []
            for value in segm_wav:
                pad = lb.util.pad_center(value,int(max_time*fs)+10)
                pad_wav.append(pad)
            
        feature_mels = []
        for value in pad_wav:
            audio = torch.from_numpy(value)
            feature = audio_class(audio)
            feature_mels.append(feature)
            
        feature_mels = pad_sequence(feature_mels, batch_first=True, padding_value=0)
        
        output_list = []
        with torch.no_grad():
            for feature_wav in feature_mels:
                feature = feature_wav.reshape(1,feature_wav.shape[1],feature_wav.shape[0])
                output = model(feature)
                #print(output)
                output = torch.round(output)
                output = output.reshape(-1).cpu().numpy()
                output_list.append(output[0])
                
        output_list = np.array(output_list)
        #print(output_list)
        preds = statistics.mode(output_list)
        #preds = preds.reshape(-1).cpu().numpy()

        category = {0:'NEGATIVE',1:'POSITIVE'}
        print('='*40)
        print('Hasil Diagnosa dari file audio '+str(filename.split('/')[-1])+' adalah')
        print(category[preds])
        print('='*40)

if __name__ == '__main__':
    fs = 16000  # Sample rate
    seconds = 3  # Duration of recording
    enablePrint()
    
    print('Preparing for Cough Record')
    time.sleep(1)
    print('...')
    time.sleep(1)
    print('...')
    time.sleep(1)
    print('...')
    
    print('Start Recording...')
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    write('output.wav', fs, myrecording)  # Save as WAV file 
    print('Stop Recording...')

    filename ='output.wav'

    predict(filename)
    time_running = round((time.time() - start_time),2)
    print("Lama diagnosa: %s seconds" %time_running )
