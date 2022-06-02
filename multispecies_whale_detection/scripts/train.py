# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model and runner for training and evaluation.

Usage templates for the command line:


BASE_DIR=$HOME/tmp/base_dir

python -m multispecies_whale_detection.scripts.train \
    --base_dir=$BASE_DIR \
    --class_names=Bm,Mn,Bp,Ej

tensorboard --logdir=$BASE_DIR/output/tensorboard


The above will use TensorFlow and Keras to train an audio event detection model,
assuming the user has set BASE_DIR to a directory they have created and in which
they have populated an input/ subdirectory. That directory should be populated,
and outputs will be written, according to a naming convention, as follows:

base_dir               (passed in via the --base_dir flag)
  +- input/
    +- train/
      +- tfrecords-*   (user-provided examplegen output to train on)
    +- validation/
      +- tfrecords-*   (user-provided examplegen output for validation)
  +- output/
    +- saved_models/   (per-epoch SavedModel exports)
    +- backup/         (checkpoint for resuming an interrupted run)
    +- tensorboard/    (logs of validation metrics to be read by TensorBoard)
"""

import os

from typing import Sequence

from absl import app
from absl import flags
import tensorflow as tf

from multispecies_whale_detection import dataset
from multispecies_whale_detection import front_end
from multispecies_whale_detection import models

FLAGS = flags.FLAGS

flags.DEFINE_string('base_dir', None,
                    'Base directory with input/ and output/ subdirectories.')
flags.DEFINE_list('class_names', None,
                  ('Label values from examplegen input CSV and output '
                   'ANNOTATION_LABEL features.'))

flags.DEFINE_integer(
    'batch_size', 384,
    'Size of minibatches, common to both training and validation.')
flags.DEFINE_float('learning_rate', 1e-4,
                   'Initial learning rate to pass to the optimizer.')
flags.DEFINE_float(
    'dropout', 0.2,
    'Fraction of units to drop out after the final global pooling layer.')

flags.DEFINE_float(
    'context_window_duration', 2.0,
    'Duration, in seconds, of audio input to a non-batch model call.')
flags.DEFINE_integer('train_windows_per_clip', 4,
                     ('Number of random-start context windows to sample '
                      'from each clip during training.'))
flags.DEFINE_integer('shuffle_buffer_size', 200000,
                     'Size of training set shuffle buffer.')


def input_filepattern(base_dir, input_subdirectory):
  """Returns an input path per convention in the module docstring."""
  return os.path.join(base_dir, 'input', input_subdirectory, 'tfrecords-*')


def probe_sample_rate(tfrecord_filepattern: str) -> int:
  """Returns the sample rate from the first example."""
  first_features = next(iter(dataset.new(tfrecord_filepattern)))
  return first_features['sample_rate'].numpy()


def main(argv: Sequence[str]) -> None:
  del argv

  base_dir = FLAGS.base_dir
  class_names = FLAGS.class_names
  batch_size = FLAGS.batch_size
  learning_rate = FLAGS.learning_rate

  def configured_window_dataset(
      input_subdirectory: str,
      windowing: dataset.Windowing,
  ) -> tf.data.Dataset:
    """Creates a Dataset, binding arguments shared by train and validation."""
    return dataset.new_window_dataset(
        tfrecord_filepattern=input_filepattern(base_dir, input_subdirectory),
        windowing=windowing,
        duration=FLAGS.context_window_duration,
        class_names=class_names,
        min_overlap=0.25,
    )

  train_dataset = configured_window_dataset(
      'train',
      dataset.RandomWindowing(FLAGS.train_windows_per_clip),
  ).cache().shuffle(FLAGS.shuffle_buffer_size).batch(
      batch_size, drop_remainder=True).prefetch(1)

  validation_dataset = configured_window_dataset(
      'validation',
      dataset.SlidingWindowing(FLAGS.context_window_duration / 2),
  ).cache().batch(
      batch_size, drop_remainder=True).prefetch(1)

  # Fail fast for empty input. (Leaving it to Keras once resulted in a cryptic
  # error message.)
  _ = next(iter(train_dataset))
  _ = next(iter(validation_dataset))

  sample_rate = probe_sample_rate(input_filepattern(base_dir, 'train'))
  context_duration_samples = int(FLAGS.context_window_duration * sample_rate)

  model = models.FramedScoreWrapper(
      layers=[
          tf.keras.Input([context_duration_samples]),
          front_end.Spectrogram(
              front_end.SpectrogramConfig(
                  sample_rate=sample_rate,
                  frame_seconds=0.05,
                  hop_seconds=0.025,
                  normalization=front_end.NoiseFloorConfig(),
                  frequency_scaling=front_end.MelScalingConfig(
                      lower_edge_hz=125.0,
                      num_mel_bins=64,
                  ))),
          front_end.SpectrogramToImage(sgram_max=30),
          tf.keras.applications.EfficientNetB0(
              include_top=False,
              weights=None,
              pooling='max',
          ),
          tf.keras.layers.Dropout(FLAGS.dropout),
          tf.keras.layers.Dense(len(class_names), activation='sigmoid'),
      ],
      input_sample_rate=sample_rate,
      class_names=class_names,
  )

  metrics = [
      tf.keras.metrics.BinaryAccuracy(),
      tf.keras.metrics.AUC(),
  ]
  for class_id, class_name in enumerate(class_names):
    metrics.extend([
        tf.keras.metrics.Precision(
            class_id=class_id, name=f'{class_name}_precision'),
        tf.keras.metrics.Recall(class_id=class_id, name=f'{class_name}_recall'),
    ])
    for recall in [0.1, 0.5, 0.8]:
      metrics.append(
          tf.keras.metrics.PrecisionAtRecall(
              recall,
              class_id=class_id,
              name=f'{class_name}_precision_at_{recall:02f}'))
    for specificity in [0.9, 0.99, 0.999]:
      metrics.append(
          tf.keras.metrics.SensitivityAtSpecificity(
              specificity,
              class_id=class_id,
              name=f'{class_name}_sensitivity_at_{specificity:02f}'))

  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
      metrics=metrics,
  )

  model.fit(
      train_dataset,
      validation_data=validation_dataset,
      epochs=100,
      #steps_per_epoch=100,
      callbacks=[
          models.SaveCallback(
              path_template=os.path.join(base_dir, 'output', 'saved_models',
                                         'epoch_{epoch:03d}'),
              signatures={
                  'score': model.score,
                  'metadata': model.metadata,
              }),
          tf.keras.callbacks.BackupAndRestore(
              os.path.join(base_dir, 'output', 'backup')),
          tf.keras.callbacks.TensorBoard(
              os.path.join(base_dir, 'output', 'tensorboard'),
              write_graph=False,
              write_steps_per_second=True,
              update_freq='epoch',
          ),
      ],
      verbose=2,  # We are using .fit() non-interactively.
  )


if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  flags.mark_flag_as_required('class_names')

  app.run(main)
