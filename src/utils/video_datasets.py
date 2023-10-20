from typing import NamedTuple, Iterator
import jax
import numpy as np 
import tensorflow_datasets as tfds

BINARIZATION_THRES = 100

class Batch(NamedTuple):
  image: jax.Array  # [B, H, W, C]
  original_image: jax.Array 
  label: int
  frame_nb: int


def extract_N_from_label(label):
  return label.split('_')[0]


def extract_frame_nb_from_filename(filename):
  return jax.numpy.array(list(map(lambda x: int(str(x).split('/')[-1].split('.')[0].split('_')[1]), filename)))

def load_dataset(path:str, split: str, batch_size: int, seed: int, binarize=False, repeat=True) -> Iterator[Batch]:
  if repeat: 
    ds = (
      tfds.ImageFolder(path).as_dataset(split=split)
        .shuffle(buffer_size=10 * batch_size, seed=seed)
        .batch(batch_size)
        .prefetch(buffer_size=5)
        .repeat()
        .as_numpy_iterator()
    )
  else: 
    ds = (
      tfds.ImageFolder(path).as_dataset(split=split)
        .shuffle(buffer_size=10 * batch_size, seed=seed)
        .batch(batch_size)
        .prefetch(buffer_size=5)
        .as_numpy_iterator()
    )
  if binarize: 
    def extract_binarized_grayscale_image(x):
        image = x['image'][...,0]
        empty_image = np.zeros_like(image)
        empty_image[image > BINARIZATION_THRES] = 1
        return Batch(image=np.expand_dims(empty_image, axis=-1), 
                    original_image=x['image'], 
                    label=x['label'],
                    frame_nb=extract_frame_nb_from_filename(x['image/filename']))
    
    return map(extract_binarized_grayscale_image, ds)
  else: 
     return map(lambda x: Batch(image=np.expand_dims(x['image'][...,0] / 255., -1),
                                original_image=x['image'], 
                                label=x['label'],
                                frame_nb=extract_frame_nb_from_filename(x['image/filename'])), 
                ds)
     


if __name__ == '__main__':
  #%%
  import matplotlib.pyplot as plt
  ds = load_dataset('data/video_datasets/2023_04_17__17_36_33', split='train', batch_size=10, seed=0)
  batch = next(ds)
  for i in range(10):
     print(batch.label)
     fig, (ax0, ax1) = plt.subplots(1,2)
     ax0.imshow(batch.original_image[i])
     ax1.imshow(batch.image[i,:,:,0])