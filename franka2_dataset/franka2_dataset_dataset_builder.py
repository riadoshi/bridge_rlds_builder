from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from franka2_dataset.conversion_utils import MultiThreadedDatasetBuilder


def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    # the line below needs to be *inside* generate_examples so that each worker creates it's own model
    # creating one shared model outside this function would cause a deadlock
    _embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _parse_examples(episode_path):
        # load raw data --> this should change for your dataset
        # data = np.load(episode_path, allow_pickle=True)  # this is a list of dicts in our case

        with tf.io.gfile.GFile(episode_path, 'rb') as f:
            data = np.load(f, allow_pickle=True)

        task_name = episode_path.split('/')[-1][:-4]
        print("task name: ", task_name)

        for k, example in enumerate(data):
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []

            for i in range(len(example['observations'])):
                observation = {
                    'state': example['observations'][i]['state'].astype(np.float32),
                }
                # orig_key = f'image'
                # new_key = f'image_0'
                # observation['image'] = example['observations'][i][orig_key]

                observation['image'] = example['observations'][i]['image']

                primitive = example['primitives'][i]

                episode.append({
                    'observation': observation,
                    'action': example['actions'][i].astype(np.float32),
                    'is_first': i == 0,
                    'is_last': i == (len(example['observations']) - 1),
                    'is_terminal': i == (len(example['observations']) - 1),
                    'language_instruction': task_name,
                    'primitive': primitive,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path,
                    'episode_id': k,
                }
            }
            
            # orig_key = f'image'
            # new_key = f'image_0'

            sample['episode_metadata'][f'has_image'] = True
            sample['episode_metadata']['has_language'] = True

            # if you want to skip an example for whatever reason, simply return None
            yield episode_path + str(k), sample

    # for smallish datasets, use single-thread parsing
    for sample in paths:
        for id, sample in _parse_examples(sample):
            yield id, sample


class Franka2Dataset(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    N_WORKERS = 40             # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 80   # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(128, 128, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc='Robot state'
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'primitive': tfds.features.Text(
                        doc='what phase the task is currently in (pick, grasp, etc)'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'episode_id': tfds.features.Scalar(
                        dtype=np.int32,
                        doc='ID of episode in file_path.'
                    ),
                    'has_image': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True if image0 exists in observation, otherwise dummy value.'
                    ),
                    'has_language': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True if language exists in observation, otherwise empty string.'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        base_path = "gs://multi-robot-bucket/multi_domain_data/franka2"
        train_filenames, val_filenames = [], []
        for filename in tf.io.gfile.glob(f'{base_path}/**/**/*.npy'):
            if 'train' in filename:
                train_filenames.append(filename)
            elif 'val' in filename:
                val_filenames.append(filename)
            else:
                raise ValueError(filename)
        print(f"Converting {len(train_filenames)} training and {len(val_filenames)} validation files.")
        return {
            'train': train_filenames,
            'val': val_filenames,
        }

