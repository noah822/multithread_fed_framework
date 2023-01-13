import torch
from PIL import Image
import numpy as np
import librosa
from torchvision import transforms
from torch.utils.data import Dataset
import os
import pandas as pd



class ravdess_dataset(Dataset):
    def __init__(self, src_path=None, csv_path=None,
                 frame_num=1,
                 image_transformer=None):
        
        '''
            src_path: source directory for `ravdess` dataset, which should have the following structure
            <src_path>
            ├── image
            └── audio
            
            frame_num: frame extracted for a given video
            image naming convention:
            <...>_<frame_index>.<extension>, where index starts from 0
            
            image_transformer: transformer for raw image data
            if not specified, default transformer is used
            
            ravdess dataset download link: https://zenodo.org/record/1188976
            including both `speech` and `sing` dataset
            for this repo, `speech` is used
            
            detailed information of ravdess dataset can be found in:
            https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
            
        '''
        
        super(ravdess_dataset, self).__init__()
        self.src_path = os.getcwd() if src_path is None else src_path
        
        if csv_path is None:
            self.csv_path = os.path.join(self.src_path, 'ravdess.csv')
        else:
            self.csv_path = csv_path
        self._csv = pd.read_csv(self.csv_path, header=None)
        
        
        
        if not os.path.exists(self.csv_path):
            self._write_csv_summary()
        
        '''
            video naming format
            
        '''
        self.frame_num = frame_num
        
        
        if image_transformer is None:
            self.image_transformer = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.image_transformer = image_transformer
            
    
    def __len__(self):
        return len(self._csv) * self.frame_num
    
    def __getitem__(self, index):
        
        _audio_path = os.path.join(self.src_path, 'audio')
        _image_path = os.path.join(self.src_path, 'image')
        
        
        
        name, label = self._csv.iloc[int(index // self.frame_num), :]
        name = name.split('.')[0]; label = torch.tensor(label)
        
        _sufix = index % self.frame_num
        
        '''
            extract audio feature with `mfcc`
            we resize the sampled data by symmetrically padding original data on both ends with 0
            the standard sample size we choose here is: 100000(sample rate: 22050)
            
            feature size: (39, 196)
        
        '''
        _standard_size = 100000


        sample, rate = librosa.load(os.path.join(_audio_path, '{}.wav'.format(name)), dtype=np.float32)
        _size = sample.shape[0]
        if _size < _standard_size:
            _padding = (_standard_size - _size) // 2
            _remainder = (_standard_size - _size) % 2
            sample = np.pad(sample, (_padding, _padding+_remainder), 'constant', constant_values=0)
        else:
            _padding = _standard_size - _size
            sample = sample[_padding, _padding + _standard_size]
        
        mfcc = librosa.feature.mfcc(y=sample, sr=rate, n_mfcc=39)
        mfcc = torch.tensor(np.expand_dims(mfcc, axis=0))
        
        

        
        img = Image.open(os.path.join(_image_path, '{}_{}.jpg'.format(name, _sufix))).convert('RGB')
        img = self.image_transformer(img)
        
        return mfcc, img, label
        
        
    
    
    def _write_csv_summary(self):
        with open(os.path.join(self.src_path, 'ravdess.csv'), 'w') as f:
            for filename in os.listdir(self.src_path):
                label = int(filename.split('.')[0].split('-')[-1])
                f.write('{},{}\n'.format(filename,label))

            
if __name__ == '__main__':
    dataset = ravdess_dataset(
        src_path='./raw/ravdess',
        frame_num=1,
    )
    
    for a, v, label in dataset:
        print(a.shape)
        print(v.shape)
        print(label)
        
