import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import stack
import numpy as np
import numpy
import pandas as pd
import random
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from scipy import signal
import wavfile
import glob

from utils.generic_utils import SpecAugmentation

class AugmentWAV(object):
    '''Credits: https://github.com/clovaai/voxceleb_trainer/blob/3bfd557fab5a3e6cd59d717f5029b3a20d22a281/DatasetLoader.py#L55'''
    def __init__(self, musan_path, noisetypes=['noise', 'speech', 'music']):

        print(noisetypes)
        self.noisetypes = noisetypes

        self.noisesnr   = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise   = {'noise':[1,1], 'speech':[3,7],  'music':[1,1] }
        self.noiselist  = {}
        augment_files   = glob.glob(os.path.join(musan_path,'*/*/*.wav'));

        for file in augment_files:
            if not file.split('/')[-3] in self.noiselist:
                self.noiselist[file.split('/')[-3]] = []
            self.noiselist[file.split('/')[-3]].append(file)
    def additive_noise(self, ap, audio, noisecat=None):
        if noisecat is None:
            augtype = random.randint(0, len(self.noisetypes))
            # if 0 dont aplly noise
            if augtype == 0:
                return audio
            noisecat = self.noisetypes[augtype-1]

        wav = audio.numpy().reshape(-1)
        clean_db = 10 * numpy.log10(numpy.mean(wav ** 2)+1e-4) 
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))

        final_noise_audio = None
        for noise in noiselist:
            noise_wav = ap.load_wav(noise).numpy().reshape(-1)
            noise_wav_len = noise_wav.shape[0]
            wav_len = wav.shape[0]
            if noise_wav_len <= wav_len:
                continue
                
            noise_start_slice = random.randint(0,noise_wav_len-(wav_len+1))
            noise_wav = noise_wav[noise_start_slice:noise_start_slice+wav_len]

            noise_snr = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noise_db = 10 * numpy.log10(numpy.mean(noise_wav ** 2)+1e-4)
            if final_noise_audio is None:
                final_noise_audio = numpy.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noise_wav
            else:
                final_noise_audio += (numpy.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noise_wav)
        if final_noise_audio is None:
            return self.additive_noise(ap, audio, noisecat)
        out_audio = torch.FloatTensor(wav + final_noise_audio).unsqueeze(0)
        return out_audio


class Dataset(Dataset):
    """
    Class for load a train and test from dataset generate by import_librispeech.py and others
    """
    def __init__(self, c, ap, train=True, max_seq_len=None, test=False, test_insert_noise=False, num_test_additive_noise=0, num_test_specaug=0):
        # set random seed
        random.seed(c.train_config['seed'])
        torch.manual_seed(c.train_config['seed'])
        torch.cuda.manual_seed(c.train_config['seed'])
        np.random.seed(c.train_config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.c = c
        self.ap = ap
        self.train = train
        self.test = test
        self.test_insert_noise = test_insert_noise
        self.num_test_additive_noise = num_test_additive_noise
        self.num_test_specaug = num_test_specaug

        self.dataset_csv = c.dataset['train_csv'] if train else c.dataset['eval_csv']
        self.dataset_root = c.dataset['train_data_root_path'] if train else c.dataset['eval_data_root_path']
        if test:
            self.dataset_csv = c.dataset['test_csv']
            self.dataset_root = c.dataset['test_data_root_path']

        assert os.path.isfile(self.dataset_csv),"Test or Train CSV file don't exists! Fix it in config.json"
        
        
        accepted_temporal_control = ['overlapping', 'padding', 'avgpool', 'speech_t', 'one_window']
        assert (self.c.dataset['temporal_control'] in accepted_temporal_control),"You cannot use the padding_with_max_length option in conjunction with the split_wav_using_overlapping option, disable one of them !!"

        self.control_class = c.dataset['control_class']
        self.patient_class = c.dataset['patient_class']

        # read csvs
        self.dataset_list = pd.read_csv(self.dataset_csv, sep=',').replace({'?': -1}).replace({'negative': self.control_class}, regex=True).replace({'positive': self.patient_class}, regex=True).values

        # get max seq lenght for padding 
        if self.c.dataset['temporal_control'] == 'padding' and train and not self.c.dataset['max_seq_len']:
            self.max_seq_len = 0
            min_seq = float('inf')
            for idx in range(len(self.dataset_list)):
                wav = self.ap.load_wav(os.path.join(self.dataset_root, self.dataset_list[idx][0]))
                # calculate time step dim using hop lenght
                seq_len = int((wav.shape[1]/c.audio['hop_length'])+1)
                if seq_len > self.max_seq_len:
                    self.max_seq_len = seq_len
                if seq_len < min_seq:
                    min_seq = seq_len
            print("The Max Time dim Lenght is: {} (+- {} seconds)".format(self.max_seq_len, ( self.max_seq_len*self.c.audio['hop_length'])/self.ap.sample_rate))
            print("The Min Time dim Lenght is: {} (+- {} seconds)".format(min_seq, (min_seq*self.c.audio['hop_length'])/self.ap.sample_rate))

        elif self.c.dataset['temporal_control'] == 'overlapping' or self.c.dataset['temporal_control'] == 'speech_t' or self.c.dataset['temporal_control'] == 'one_window':
            # set max len for window_len seconds multiply by sample_rate and divide by hop_lenght
            self.max_seq_len = int(((self.c.dataset['window_len']*self.ap.sample_rate)/c.audio['hop_length'])+1)
            print("The Max Time dim Lenght is: ", self.max_seq_len, "It's use overlapping technique, window:", self.c.dataset['window_len'], "step:", self.c.dataset['step'])
        else: # for eval set max_seq_len in train mode
            if self.c.dataset['max_seq_len']:
                self.max_seq_len = self.c.dataset['max_seq_len']
            else:
                self.max_seq_len = max_seq_len
        
        if self.c.data_aumentation['insert_noise']:
            self.augment_wav = AugmentWAV(self.c.data_aumentation['musan_path'], noisetypes=self.c.data_aumentation['noisetypes'])
        else:
            self.augment_wav = None

        if self.test_insert_noise:
            self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=8, freq_stripes_num=2)
            self.augment_wav = AugmentWAV(self.c.data_aumentation['musan_path'], noisetypes=self.c.data_aumentation['noisetypes'])
            if not self.num_test_additive_noise and not self.num_test_specaug:
                raise RuntimeError("ERROR: when  test_insert_noise is True, num_test_additive_noise or  num_test_specaug need to be > 0")
            if self.c.dataset['temporal_control'] == 'overlapping' or self.c.dataset['temporal_control'] == 'speech_t':
                raise RuntimeError("ERROR: Noise insertion in  'temporal_control' overlapping and  speech_t is not supported !! You need implement it !!")
        else:
            self.spec_augmenter = None

    def get_max_seq_lenght(self):
        return self.max_seq_len

    def __getitem__(self, idx):
        wavfile_name = self.dataset_list[idx][0]
        wav = self.ap.load_wav(os.path.join(self.dataset_root, wavfile_name))
        #print("FILE:", os.path.join(self.dataset_root, self.dataset_list[idx][0]), wav.shape)
        class_name = self.dataset_list[idx][1]
        # print('class before transform', class_name)
        if str(class_name) == 'positive':
           class_name = self.patient_class
        elif str(class_name) == 'negative':
            class_name = self.control_class

        # print('class after transform',class_name)
        # its assume that noise file is biggest than wav file !!
        # torchaudio.save('wav.wav', wav, self.ap.sample_rate)   
        if self.c.data_aumentation['insert_noise'] and self.train:
            wav = self.augment_wav.additive_noise(self.ap, wav)

        if self.c.dataset['temporal_control'] == 'overlapping' or self.c.dataset['temporal_control'] == 'speech_t':
            #print("Wav len:", wav.shape[1])
            #print("for",self.ap.sample_rate*self.c.dataset['window_len'], wav.shape[1], self.ap.sample_rate*self.c.dataset['step'])
            start_slice = 0
            features = []
            targets = []
            step = self.ap.sample_rate*self.c.dataset['step']
            for end_slice in np.arange(self.ap.sample_rate*self.c.dataset['window_len'], wav.shape[1], step):
                # print("Slices: ", start_slice, end_slice)
                start_slice = int(start_slice)
                end_slice = int(end_slice)
                spec = self.ap.get_feature_from_audio(wav[:, start_slice:end_slice]).transpose(1, 2)
                #print(np.array(spec).shape)
                features.append(spec)
                targets.append(torch.FloatTensor([class_name]))
                start_slice += step
            if len(features) == 1:
                feature = self.ap.get_feature_from_audio(wav[:, :self.ap.sample_rate*self.c.dataset['window_len']]).transpose(1, 2)
                target = torch.FloatTensor([class_name])
            elif len(features) < 1:
                # padding audios less than 2 seconds
                feature = self.ap.get_feature_from_audio(wav).transpose(1, 2)
                # padding for max sequence 
                zeros = torch.zeros(1, self.max_seq_len - feature.size(1), feature.size(2))
                # append zeros before features
                feature = torch.cat([feature, zeros], 1)
                target = torch.FloatTensor([class_name])
                #print("ERROR: Some sample in your dataset is less than %d seconds! Change the size of the overleapping window"%self.c.dataset['window_len'])
                #raise RuntimeError("ERROR: Some sample in your dataset is less than {} seconds! Change the size of the overleapping window (CONFIG.dataset['window_len'])".format(self.c.dataset['window_len']))
            else:
                feature = torch.cat(features, dim=0)
                target = torch.cat(targets, dim=0)
            if self.c.dataset['temporal_control'] == 'speech_t':
                feature = feature.unsqueeze(1)                
                target = torch.FloatTensor([target[0]])

        elif self.c.dataset['temporal_control'] == 'one_window':
            # print( "one_window")
            # choise a random part of audio 
            step = self.ap.sample_rate*self.c.dataset['window_len']
            # print(wav.shape, step)
            idx = random.randint(0, wav.size(1)-(step+1))
            feature = self.ap.get_feature_from_audio(wav[:, idx:idx+step]).transpose(1, 2)
            target = torch.FloatTensor([class_name])
        else:
            # feature shape (Batch_size, n_features, timestamp)
            feature = self.ap.get_feature_from_audio(wav)
            # transpose for (Batch_size, timestamp, n_features)
            feature = feature.transpose(1,2)
            # remove batch dim = (timestamp, n_features)
            feature = feature.reshape(feature.shape[1:])
            if self.c.dataset['temporal_control'] == 'padding':
                # padding for max sequence 
                zeros = torch.zeros(self.max_seq_len - feature.size(0),feature.size(1))
                # append zeros before features
                feature = torch.cat([feature, zeros], 0)
                target = torch.FloatTensor([class_name])                
            else: # avgpoling
                target = torch.FloatTensor([class_name])


        if self.test_insert_noise and not self.c.dataset['temporal_control'] == 'overlapping' and not self.c.dataset['temporal_control'] == 'speech_t':
            features = []
            targets = []
            feature = feature.unsqueeze(0)
            target = target.unsqueeze(0)
            features.append(feature)
            targets.append(target)
            for _ in range(self.num_test_additive_noise):
                wav_noise = self.augment_wav.additive_noise(self.ap, wav)
                feature = self.ap.get_feature_from_audio(wav_noise)
                # transpose for (Batch_size, timestamp, n_features)
                feature = feature.transpose(1, 2)
                if self.c.dataset['temporal_control'] == 'padding':
                    # padding for max sequence 
                    zeros = torch.zeros(feature.size(0), self.max_seq_len - feature.size(1), feature.size(2))
                    # append zeros before features
                    feature = torch.cat([feature, zeros], 1)
                # print(feature.shape)
                features.append(feature)
                targets.append(target)

            # print(len(features))
            feature = torch.cat(features, dim=0)
            target = torch.cat(targets, dim=0)
            
            # print("Additive:", feature.shape, target.shape, self.max_seq_len)
            if self.num_test_specaug:
                features_aug = self.ap.get_feature_from_audio(wav)
                # transpose for (Batch_size, timestamp, n_features)
                features_aug = features_aug.transpose(1,2)
                # repeat tensor because its more fast than simply append on a list
                targets_aug = torch.FloatTensor([class_name]).repeat(self.num_test_specaug, 1)
                features_aug = features_aug.repeat(self.num_test_specaug, 1, 1)
                # apply spec augmentation in the features
                features_aug = self.spec_augmenter(features_aug.unsqueeze(1), test=True).squeeze(1)
                if self.c.dataset['temporal_control'] == 'padding':
                    # padding for max sequence 
                    zeros = torch.zeros(features_aug.size(0), self.max_seq_len - features_aug.size(1),features_aug.size(2))
                    # append zeros before features
                    features_aug = torch.cat([features_aug, zeros], 1)
                # print("end..:", features_aug.shape, targets_aug.shape)
                
                # concate new features
                feature = torch.cat([feature, features_aug], dim=0)
                target = torch.cat([target, targets_aug], dim=0)
                # target = torch.FloatTensor([class_name]).repeat(feature.size(0), 1)
        # print("After Spec:", feature.shape, target.shape)    
        if self.test:
            return feature, target, wavfile_name
        return feature, target

    def __len__(self):
        return len(self.dataset_list)

def train_dataloader(c, ap, class_balancer_batch=False):
    dataset = Dataset(c, ap, train=True)
    if class_balancer_batch:
        shuffle=False
        print("Using Class Batch Balancer")
        classes_list = [cl[1] for cl in dataset.dataset_list]

        classes_list = np.array(classes_list)
        
        unique_class_names = np.unique(classes_list).tolist()
        class_ids = [unique_class_names.index(s) for s in classes_list]

        # count number samples by class
        class_count = np.array([len(np.where(classes_list == c)[0]) for c in unique_class_names])
        
        # create weight
        weight = 1. / class_count
        samples_weight = np.array([weight[c] for c in class_ids])
        
        class_dataset_samples_weight = torch.from_numpy(samples_weight).double()
        # create sampler
        sampler = torch.utils.data.sampler.WeightedRandomSampler(class_dataset_samples_weight, len(class_dataset_samples_weight), replacement=True)
    else: 
         sampler=None
         shuffle=True
    return DataLoader(dataset=dataset,
                          batch_size=c.train_config['batch_size'],
                          shuffle=shuffle,
                          num_workers=c.train_config['num_workers'],
                          collate_fn=own_collate_fn,
                          pin_memory=True,
                          drop_last=True,
                          sampler=sampler)

def eval_dataloader(c, ap, max_seq_len=None):
    return DataLoader(dataset=Dataset(c, ap, train=False, max_seq_len=max_seq_len),
                          collate_fn=val_collate_fn, batch_size=c.test_config['batch_size'], 
                          shuffle=False, num_workers=c.test_config['num_workers'])


def test_dataloader(c, ap, max_seq_len=None, insert_noise=False, num_additive_noise=0, num_specaug=0):
    return DataLoader(dataset=Dataset(c, ap, train=False, test=True, max_seq_len=max_seq_len, test_insert_noise=insert_noise, num_test_additive_noise=num_additive_noise, num_test_specaug=num_specaug),
                          collate_fn=teste_collate_fn, batch_size=c.test_config['batch_size'], 
                          shuffle=False, num_workers=c.test_config['num_workers'])

def own_collate_fn(batch):
    features = []
    targets = []
    for feature, target in batch:
        features.append(feature)
        #print(target.shape)
        targets.append(target)
    
    if len(features[0].shape) == 3: # if dim is 3, we have a many specs because we use a overlapping
        targets = torch.cat(targets, dim=0)
        features = torch.cat(features, dim=0)
    elif len(features[0].shape) == 4: # if dim = 4 is speech transformer mode
        features = pad_sequence(features, batch_first=True, padding_value=0).squeeze(2)
        targets = torch.cat(targets, dim=0)
    else:    
        # padding with zeros timestamp dim
        features = pad_sequence(features, batch_first=True, padding_value=0)
        # its padding with zeros but mybe its a problem because 
        targets = pad_sequence(targets, batch_first=True, padding_value=0)

    # 
    targets = targets.reshape(targets.size(0), -1)
    # list to tensor
    #targets = stack(targets, dim=0)
    #features = stack(features, dim=0)
    #print(features.shape, targets.shape)
    return features, targets


def val_collate_fn(batch):
    features = []
    targets = []
    slices = []
    targets_org = []
    for feature, target in batch:
        features.append(feature)
        targets.append(target)
        if len(feature.shape) == 3:
            slices.append(torch.tensor(feature.shape[0]))
            # its used for integrity check during unpack overlapping for calculation loss and accuracy
            targets_org.append(target[0])
    
    if len(features[0].shape) == 3: # if dim is 3, we have a many specs because we use a overlapping
        targets = torch.cat(targets, dim=0)
        features = torch.cat(features, dim=0)
    elif len(features[0].shape) == 4: # if dim = 4 is speech transformer mode
        features = pad_sequence(features, batch_first=True, padding_value=0).squeeze(2)
        targets = torch.cat(targets, dim=0)
    else:    
        # padding with zeros timestamp dim
        features = pad_sequence(features, batch_first=True, padding_value=0)
        # its padding with zeros but mybe its a problem because 
        targets = pad_sequence(targets, batch_first=True, padding_value=0)

    # 
    targets = targets.reshape(targets.size(0), -1)

    if slices:
        slices = stack(slices, dim=0)
        targets_org = stack(targets_org, dim=0)
    else:
        slices = None
        targets_org = None
    # list to tensor
    #targets = stack(targets, dim=0)
    #features = stack(features, dim=0)
    #print(features.shape, targets.shape)
    return features, targets, slices, targets_org


def teste_collate_fn(batch):
    features = []
    targets = []
    slices = []
    targets_org = []
    names = []
    for feature, target, file_name in batch:
        features.append(feature)
        targets.append(target)
        names.append(file_name)
        if len(feature.shape) == 3:
            slices.append(torch.tensor(feature.shape[0]))
            # its used for integrity check during unpack overlapping for calculation loss and accuracy
            targets_org.append(target[0])
    
    if len(features[0].shape) == 3: # if dim is 3, we have a many specs because we use a overlapping
        targets = torch.cat(targets, dim=0)
        features = torch.cat(features, dim=0)
    elif len(features[0].shape) == 4: # if dim = 4 is speech transformer mode
        features = pad_sequence(features, batch_first=True, padding_value=0).squeeze(2)
        targets = torch.cat(targets, dim=0)
    else:    
        # padding with zeros timestamp dim
        features = pad_sequence(features, batch_first=True, padding_value=0)
        # its padding with zeros but mybe its a problem because 
        targets = pad_sequence(targets, batch_first=True, padding_value=0)

    # 
    targets = targets.reshape(targets.size(0), -1)

    if slices:
        slices = stack(slices, dim=0)
        targets_org = stack(targets_org, dim=0)
    else:
        slices = None
        targets_org = None
    # list to tensor
    #targets = stack(targets, dim=0)
    #features = stack(features, dim=0)
    #print(features.shape, targets.shape)
    return features, targets, slices, targets_org, names