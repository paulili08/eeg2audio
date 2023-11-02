import os
import torch
import pandas as pd
import numpy as np 
import numpy.matlib as matlib
import scipy
import scipy.signal
import scipy.stats
import scipy.io.wavfile
import scipy.fftpack
# Custom dataset
from PIL import Image
import torch.utils.data as data
import os
import random

from pynwb import NWBHDF5IO
import MelFilterBank as mel

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, subfolder='train', transform=None, resize_scale=None, crop_size=None, fliplr=False, is_color=True):
        super(DatasetFromFolder, self).__init__()
        self.input_path = os.path.join(image_dir, subfolder)
        self.image_filenames = [x for x in sorted(os.listdir(self.input_path))]
        self.transform = transform
        self.resize_scale = resize_scale
        self.crop_size = crop_size
        self.fliplr = fliplr
        self.is_color = is_color

    def __getitem__(self, index):
        # Load Image
        img_fn = os.path.join(self.input_path, self.image_filenames[index])
        if self.is_color:
            img = Image.open(img_fn).convert('RGB')
        else:
            img = Image.open(img_fn)

        # preprocessing
        if self.resize_scale is not None:
            img = img.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)

        if self.crop_size is not None:
            x = random.randint(0, self.resize_scale - self.crop_size + 1)
            y = random.randint(0, self.resize_scale - self.crop_size + 1)
            img = img.crop((x, y, x + self.crop_size, y + self.crop_size))
        if self.fliplr:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.image_filenames)

class EEGDataset(data.Dataset):
    def __init__(self, eeg_data):
        super(EEGDataset, self).__init__()
        self.eeg_data = eeg_data

    def __getitem__(self, idx):
        eeg = self.eeg_data[idx]
        return torch.Tensor(eeg)

    def __len__(self):
        return len(self.eeg_data)



if __name__=="__main__":
    winL = 0.05
    frameshift = 0.01
    modelOrder = 4
    stepSize = 5
    path_bids = r'./SingleWordProductionDutch-iBIDS'
    path_output = r'./features'
    participants = pd.read_csv(os.path.join(path_bids,'participants.tsv'), delimiter='\t')
    for p_id, participant in enumerate(participants['participant_id']):
        
        #Load data
        io = NWBHDF5IO(os.path.join(path_bids,participant,'ieeg',f'{participant}_task-wordProduction_ieeg.nwb'), 'r')
        nwbfile = io.read()
        #sEEG
        eeg = nwbfile.acquisition['iEEG'].data[:]
        eeg_sr = 1024
        #audio
        audio = nwbfile.acquisition['Audio'].data[:]
        audio_sr = 48000
        #words (markers)
        words = nwbfile.acquisition['Stimulus'].data[:]
        words = np.array(words, dtype=str)
        io.close()
        #channels
        channels = pd.read_csv(os.path.join(path_bids,participant,'ieeg',f'{participant}_task-wordProduction_channels.tsv'), delimiter='\t')
        channels = np.array(channels['name'])
        
        #Process Audio
        target_SR = 16000
        audio = scipy.signal.decimate(audio,int(audio_sr / target_SR))
        audio_sr = target_SR
        scaled = np.int16(audio/np.max(np.abs(audio)) * 32767)
        os.makedirs(os.path.join(path_output), exist_ok=True)
        scipy.io.wavfile.write(os.path.join(path_output,f'{participant}_orig_audio.wav'),audio_sr,scaled)   

        