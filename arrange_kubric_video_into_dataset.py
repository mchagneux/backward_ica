#%%
from typing import Iterator, Tuple, NamedTuple, Sequence
import jax 
import os 
import tensorflow_datasets as tfds
import shutil 
from datetime import datetime
import numpy as np
from src.utils.misc import Serializer
import dataclasses

class Batch(NamedTuple):
  image: jax.Array  # [B, H, W, C]


date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
dataset_path = os.path.join('datasets', date)
os.makedirs(dataset_path, exist_ok=True)
serializer = Serializer(dataset_path)


@dataclasses.dataclass
class Config:
  type: str = ''
  video_path:str = ''
  train_video_path: str = ''
  valid_video_path: str = ''
  train_proportion: float = -1





def one_video_per_split(train_video_path, valid_video_path):
  serializer.save_config(config=Config(type='one_video_per_split', 
                                       train_video_path=train_video_path, 
                                       valid_video_path=valid_video_path))
  
  for video_path, split in zip([train_video_path, valid_video_path], 
                              ['train', 'valid']):
    images_path = os.path.join(video_path, 'images')
    Ns = np.load(os.path.join(video_path, 'N.npy'))
    # N_pluses = np.load(os.path.join(video_path, 'N_plus.npy'))
    # N_minuses = np.load(os.path.join(video_path, 'N_minus.npy'))

    image_names = [image_name for image_name in  os.listdir(images_path) if 'rgba' in image_name]

    for image_name in image_names:
      image_nb = int(image_name.split('_')[1].split('.')[0])
      N = int(Ns[image_nb]-1)
      # N_plus = int(N_pluses[image_nb])
      # N_minus = int(N_minuses[image_nb])

      image_target_dir = os.path.join(dataset_path, split, f'{N}') #_{N_plus}_{N_minus}')
      os.makedirs(image_target_dir, exist_ok=True)
      shutil.copy(os.path.join(images_path, image_name), 
                  os.path.join(image_target_dir, image_name))
      
def one_video_shuffled_into_two_splits(video_path, train_proportion):
  serializer.save_config(config=Config(type='one_video_shuffle_into_two_splits', 
                                       video_path=video_path, 
                                       train_proportion=train_proportion))
  
  
  
  images_path = os.path.join(video_path, 'images')
  Ns = np.load(os.path.join(video_path, 'N.npy'))

  images_names = np.random.permutation([image_name for image_name in \
                        os.listdir(images_path) if 'rgba' in image_name])

  train_valid_jump = int(train_proportion*len(images_names))
  frames = {'train':images_names[:train_valid_jump], 
            'valid':images_names[train_valid_jump:]}

  for split in ['train', 'valid']:
    for image_name in frames[split]:
      image_nb = int(image_name.split('_')[1].split('.')[0])
      N_image = int(Ns[image_nb]-1)
      image_target_dir = os.path.join(dataset_path, split, str(N_image))
      os.makedirs(image_target_dir, exist_ok=True)
      shutil.copy(os.path.join(images_path, image_name), 
                  os.path.join(image_target_dir, image_name))
    
if __name__ == '__main__':
  one_video_per_split(train_video_path='kubric/output/2023_04_17__14_55_24/301', 
                      valid_video_path='kubric/output/2023_04_17__15_18_38/289')
#%%



