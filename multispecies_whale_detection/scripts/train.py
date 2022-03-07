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
"""

from typing import Sequence

from absl import app
from absl import flags
import tensorflow as tf

from multispecies_whale_detection import dataset

FLAGS = flags.FLAGS

flags.DEFINE_string('base_dir', None, 'Base directory for input and output.')


def main(argv: Sequence[str]) -> None:
  del argv
  print(FLAGS.base_dir)


if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  app.run(main)
