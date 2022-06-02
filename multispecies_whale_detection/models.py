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
"""Keras model adapters and components."""
import tensorflow as tf


class FramedScoreWrapper(tf.keras.Sequential):
  """Wrapper for a window-level Model to frame and score longer clips."""

  def __init__(self, layers, input_sample_rate, class_names):
    super(FramedScoreWrapper, self).__init__(layers=layers)
    self._context_width_samples = layers[0].shape[-1]
    self._input_sample_rate = input_sample_rate
    self._class_names = class_names

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
      tf.TensorSpec(shape=tuple(), dtype=tf.int64)
  ])
  def score(self, waveform, context_step_samples):
    waveform = tf.squeeze(waveform, -1)  # ensures single-channel
    batch_size = tf.shape(waveform)[0]
    context_step_samples = tf.cast(context_step_samples, tf.int32)
    context_windows = tf.signal.frame(
        waveform, self._context_width_samples, context_step_samples, axis=1)
    num_windows = tf.shape(context_windows)[1]
    waveform_batch = tf.reshape(context_windows,
                                [-1, self._context_width_samples])
    scores = self(waveform_batch)
    return {
        'scores':
            tf.reshape(scores,
                       [batch_size, num_windows,
                        len(self._class_names)])
    }

  @tf.function(input_signature=[])
  def metadata(self):
    return {
        'input_sample_rate': self._input_sample_rate,
        'context_width_samples': self._context_width_samples,
        'class_names': tf.constant(self._class_names),
    }


class SaveCallback(tf.keras.callbacks.Callback):
  """Callback to export SavedModels using the Keras API.

  The purpose is the same as that of tf.keras.callbacks.ModelCheckpoint, but
  that does not support setting a custom value of signatures=, so we create
  a special-case version here.
  """

  def __init__(self, path_template, signatures):
    """Initializes this callback.

    Args:
      path_template: Formatting pattern that may include {epoch} to keep
        separate outputs for each epoch.
      signatures: See tf.saved_model.save, to which this argument is forwarded.
    """
    self._path_template = path_template
    self._signatures = signatures

  def on_epoch_end(self, epoch, logs=None):
    """Calls keras.Model.save with custom signatures."""
    self.model.save(
        self._path_template.format(epoch=epoch), signatures=self._signatures)
