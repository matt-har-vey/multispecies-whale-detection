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

python -m multispecies_whale_detection.scripts.train

will use TensorFlow and Keras to train an audio event detection model and
periodically log evaluation metrics.

base_dir
  +- input/
    +- train/
      +- tfrecords-*
    +- validation/
      +- tfrecords-*
  +- output/
"""

import os

from typing import Sequence

from absl import app
from absl import flags
import tensorflow as tf

from multispecies_whale_detection import dataset

FLAGS = flags.FLAGS

flags.DEFINE_string('base_dir', None,
                    'Base directory with input/ and output/ subdirectories.')

CLASS_NAMES = ['Orca', 'SRKW']


def main(argv: Sequence[str]) -> None:
  del argv

  batch_size = 512

  train_dataset = dataset.new_window_dataset(
      tfrecord_filepattern=os.path.join(FLAGS.base_dir, 'input', 'train',
                                        'tfrecords-*'),
      duration=1.0,
      class_names=CLASS_NAMES,
      windowing=dataset.RandomWindowing(4),
      min_overlap=0.25,
  ).shuffle(batch_size * 4).batch(batch_size).prefetch(1)

  validation_dataset = dataset.new_window_dataset(
      tfrecord_filepattern=os.path.join(FLAGS.base_dir, 'input', 'validation',
                                        'tfrecords-*'),
      duration=1.0,
      class_names=CLASS_NAMES,
      windowing=dataset.RandomWindowing(4),
      min_overlap=0.25,
  ).batch(batch_size).prefetch(1)

  for batch in validation_dataset:
    features, labels = batch
    if tf.reduce_any(labels > 0.0):
      print(batch)
      break


if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  app.run(main)
