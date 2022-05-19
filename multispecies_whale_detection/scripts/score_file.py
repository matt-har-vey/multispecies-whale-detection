""" Utility to score a SavedModel on all FLAC files in a given directory."""

import os
import csv

from typing import Sequence

from absl import app
from absl import flags
import resampy
import soundfile
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('saved_model', None, 'model to load.')
flags.DEFINE_string('audio_dir', None, 'directory of files to analyze.')
flags.DEFINE_integer('sample_rate', None, 'sample rate expected by the model.')
flags.DEFINE_float('window_duration', 1.0, 'duration of model input.')
flags.DEFINE_float('min_score', None, 'minimum score to include in output.')
flags.DEFINE_string('output_file', None, 'name of output file.')

class_names = ['Orca','SRKW','J','K','L','NRKW','IBKW','HUMP','BG']

def main(argv: Sequence[str]) -> None:
  del argv
  model = tf.keras.models.load_model(FLAGS.saved_model)
  score_outfile = open(os.path.join((FLAGS.audio_dir), os.path.basename(FLAGS.saved_model)+'_'+FLAGS.output_file+'.csv'), 'w', newline="")
  csv_out = csv.writer(score_outfile)
  for filename in os.listdir(FLAGS.audio_dir):
    print(filename)
    if not filename.endswith('.flac'):
      continue
    full_filename = os.path.join(FLAGS.audio_dir, filename)
    data, original_sample_rate = soundfile.read(full_filename)
    resampled_audio = resampy.resample(data, original_sample_rate,
                                       FLAGS.sample_rate)
    window_duration_samples = int(FLAGS.window_duration * FLAGS.sample_rate)
    model_input = tf.signal.frame(
        signal=resampled_audio,
        frame_length=window_duration_samples,
        frame_step=window_duration_samples,
    )
    csv_out.writerow([full_filename])
    scores = model(model_input)
    row_output = [] 
    for seconds, row in enumerate(scores):
      whale_type_scores = tf.unstack(row)
      class_count = 0
      row_output.append(seconds)

      for whale_type_score in whale_type_scores:
      
        if whale_type_score.numpy() > FLAGS.min_score:
          row_output.append(class_names[class_count])
          row_output.append(whale_type_score.numpy())
        class_count = class_count + 1
      
      if len(row_output) > 1:
        csv_out.writerow(row_output)
      row_output.clear()
  score_outfile.close()


if __name__ == '__main__':
  flags.mark_flag_as_required('saved_model')
  flags.mark_flag_as_required('audio_dir')
  flags.mark_flag_as_required('sample_rate')
  flags.mark_flag_as_required('min_score')
  flags.mark_flag_as_required('output_file')

  app.run(main)
