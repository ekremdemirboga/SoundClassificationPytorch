from torch.utils.data import Dataset
import torch
import pandas as pd
import torchaudio
import os


class UrbanSoundDataset(Dataset):
    
    def __init__(self,
                 annotation_file, 
                 audio_dir, 
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):
        self.annotations = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal,sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal) ## if the number of samples is less then we want
        signal = self._cut_if_necessary(signal) ## if the number of samples is more then we want
        signal = self.transformation(signal)
        return signal, label
        ## signal -> Tensor(numofchannels=1, numsamples=sum_samples)
    def _cut_if_necessary(self, signal):
        if signal.shape[1]>self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples-length_signal
            last_dim_padding = (0, num_missing_samples) ##0 to left, num_missing_samples to right
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    
    def _resample_if_necessary(self, signal, sr): 
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0]>1:
            signal = torch.mean(signal, dim=0, keepdims=True)
        return signal
    
    def _get_audio_sample_path(self, index):
        fold = f"fold{self.annotations.iloc[index,5]}"
        path = os.path.join(self.audio_dir,fold,self.annotations.iloc[index,0])
        return path
    
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index,6]
    
    ##a_list[1] -> a_list.__getitem__(1)


prefix = "C:/Users/ekrem/Desktop/makamFinder/pytorch_test/audio_test/Data/UrbanSound8K/"
ANNOTATIONS_FILE = prefix + "metadata/UrbanSound8K.csv"
AUDIO_DIR = prefix + "audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050 ## it means one seconds of audio

if torch.cuda.is_available():
    device ="cuda"
else:
    device ="cpu"
print("using device :" + str(device))

mel_spectogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

usd = UrbanSoundDataset (ANNOTATIONS_FILE, AUDIO_DIR, mel_spectogram,
                        SAMPLE_RATE,NUM_SAMPLES,device)

print(f"there are {len(usd)} samples in the dataset")

#signal, label = usd[1]






