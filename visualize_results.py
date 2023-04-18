#%%
import matplotlib.pyplot as plt 
from src.variational.inference_nets import build_model
from src.video_datasets import load_dataset, Batch
from src.utils import Serializer
import jax 
import numpy as np
import haiku as hk
from src.utils import classify_fn

nb_images_to_show = 10
def visualize(vae_path=None, classif_path=None, dataset_path=None):

    if classif_path is not None: 
        serializer_classif = Serializer(classif_path)
        config_classif = serializer_classif.load_config()
        classifier_weights = serializer_classif.load_params()
        serializer_vae = Serializer(config_classif['pretrained_vae_path'])
        config_vae = serializer_vae.load_config()
        classifier = hk.without_apply_rng(hk.transform(classify_fn))


    if vae_path is not None: 
        serializer_vae = Serializer(vae_path)
        config_vae = serializer_vae.load_config()

    if dataset_path is None: 
        dataset_path = config_vae['dataset_path']

    dataset = load_dataset(dataset_path, 'train', nb_images_to_show, 0)
    key = jax.random.PRNGKey(0)

    pretrained_vae_weights = serializer_vae.load_params()
    vae_model = build_model(config=config_vae)

    batch = next(dataset)
    output_vae = vae_model.apply(pretrained_vae_weights, key, batch.image)

    if classif_path is not None: 
        output_classif = jax.nn.softmax(classifier.apply(classifier_weights, output_vae.mean_z))


    for i in range(nb_images_to_show):
        fig, (ax0, ax1, ax2) = plt.subplots(1,3)
        ax0.imshow(batch.original_image[i,:,:])
        ax0.set_title('Original image')
        ax1.imshow((batch.image[i,:,:,0]))
        ax1.set_title('Dataset image')
        reconstructed_image = np.array(output_vae.image[i,:,:,0])
        ax2.imshow(reconstructed_image)
        ax2.set_title('Reconstructed image')
        if classif_path is not None:
            plt.suptitle(f'Predicted nb of objects: {np.argmax(output_classif[i])+1}')
        plt.tight_layout()
        plt.autoscale(True)


visualize(classif_path='experiments/latent_classif_training/2023_04_18__09_40_59')
#%%