#%%
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# image = mnist_dataset.next()['image']
moving_mnist = np.load('data/moving_mnist/mnist_test_seq.npy')

#%%
# sequence = tfds.video.moving_mnist.image_as_moving_sequence(image).image_sequence

# for image in sequence: 
#     plt.imshow(image, cmap='gray')
#     plt.show()