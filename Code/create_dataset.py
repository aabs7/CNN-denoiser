from data_processing.dataset import Dataset
from data_processing.urban_sound import UrbanSound8K
from data_processing.mozilla_common_voice import MozillaCommonVoiceDataset
import warnings

warnings.filterwarnings(action='ignore')

mozilla_basepath = '/home/abhish/Documents/Dataset/MozillaDataset'
urbansound_basepath = '/home/abhish/Documents/Dataset/UrbanSound8K'

# load noise audio train and validation filenames
us8k = UrbanSound8K(urbansound_basepath,val_dataset_size = 200)
noise_train_filenames,noise_val_filenames = us8k.get_train_val_filenames()

# load clean audio train and validation filenames
mcv = MozillaCommonVoiceDataset(mozilla_basepath,val_dataset_size = 1000)
clean_train_filenames,clean_val_filenames = mcv.get_train_val_filenames()


# Parameters about window and audio
windowLength = 256
config = {'windowLength': windowLength,
          'overlap': round(0.25 * windowLength),
          'fs': 16000,
          'audio_max_duration': 0.8}

## create train and validation dataset from audio dataset. 

val_dataset = Dataset(clean_val_filenames,noise_val_filenames,**config)
val_dataset.create_tf_record(prefix = 'val', subset_size = 2000)
train_dataset = Dataset(clean_train_filenames,noise_train_filenames, **config)
train_dataset.create_tf_record(prefix = 'train', subset_size = 2000)

## create test set
clean_test_filenames = mcv.get_test_filenames()
noise_test_filenames = us8k.get_test_filenames()

test_dataset = Dataset(clean_test_filenames,noise_test_filenames,**config)
test_dataset.create_tf_record(prefix = 'test',subset_size = 1000, parallel = False)