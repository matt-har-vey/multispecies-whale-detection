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
"""Runner and configuation for the examplegen Beam pipeline.

Usage:
  python3 run_examplegen_local.py

The configuration hard-codes subdirectories of the user's home directory for
input and output. This file is intended as configuration-only, as an alternative
to a long command line. Configuration changes should be made by editing it
directly.
"""
import os
import pathlib
import shutil

from apache_beam.options import pipeline_options

from multispecies_whale_detection import examplegen


def run():
  """Configures and runs the pipeline and waits on completion."""
  configuration = examplegen.Configuration(
      input_directory=os.path.expanduser('~/tmp/examplegen/input'),
      output_directory=os.path.expanduser('~/tmp/examplegen/output'),
      resample_rate=24000,
      clip_duration_seconds=2.0,
  )

  options = pipeline_options.PipelineOptions(runner='SparkRunner')
  examplegen.run(configuration, options)


if __name__ == '__main__':
  run()
