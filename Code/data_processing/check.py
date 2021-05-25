from data_processing.urban_sound import UrbanSound8K
from data_processing.feature_extractor import FeatureExtractor
from data_processing.dataset import Dataset
from data_processing.mozilla_common_voice import MozillaCommonVoiceDataset
from utils import prepare_input_features, read_audio
import numpy as np
import matplotlib.pyplot as plt
import librosa

windowLength = 256
config = {'windowLength': windowLength,
          'overlap': round(0.25 * windowLength),
          'fs': 16000,
          'audio_max_duration': 0.8}

urbansound_basepath = '/home/abhish/Documents/Dataset/UrbanSound8K'
mozilla_basepath = '/home/abhish/Documents/Dataset/MozillaDataset'

mcv = MozillaCommonVoiceDataset(mozilla_basepath,val_dataset_size = 100)
clean_train_filenames,clean_val_filenames = mcv.get_train_val_filenames()

us8k = UrbanSound8K(urbansound_basepath,val_dataset_size = 100)
noise_train_filenames,noise_val_filenames = us8k.get_train_val_filenames()

#clean_filename = np.random.choice(clean_train_filenames)
#noise_val_filename = np.random.choice(noise_val_filenames)


clean_dataset = Dataset(clean_train_filenames, noise_train_filenames, **config)

noise_mag,clean_mag,noise_phase = clean_dataset.return_values()

print("Noise mag shape:",noise_mag.shape)

stftSegment = prepare_input_features(noise_mag,numSegments=8,numFeatures=129)

noise_stft_mag_features = np.transpose(stftSegment,(2,0,1))
clean_mag = np.transpose(clean_mag,(1,0))
noise_phase = np.transpose(noise_phase,(1,0))

noise_stft_mag_features = np.expand_dims(noise_stft_mag_features,axis = 3)
clean_mag = np.expand_dims(clean_mag,axis = 2)
count = 0

# for x_,y_,p_ in zip(noise_stft_mag_features,clean_mag,noise_phase):
#     y_ = np.expand_dims(y_,2)
#     print(x_.shape)
#     print(y_.shape)
#     print(p_.shape)
#     count += 1
# print(count)