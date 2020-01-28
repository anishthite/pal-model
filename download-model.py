#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
#
# Please assign the DATA_FOLDER before running this scripts, the data, pre-trained model, fine-tuned model will be
# downloaded automatically to DATA_FOLDER

import os
import sys
import logging
from functools import partial

from demo_utils import download_model_folder
import argparse
import subprocess as sp


PROJECT_FOLDER = os.path.dirname(os.path.realpath(__file__))
PYTHON_EXE = 'python'
MODEL_FOLDER = os.path.join(PROJECT_FOLDER, 'models')
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')

print(f'PROJECT_FOLDER = {PROJECT_FOLDER}')

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='dummy',
                    help='choose from dummy, small and full')
dargs = parser.parse_args()

assert dargs.data == 'dummy' or dargs.data == 'small' or dargs.data == 'full' , \
    'The specified data option is not support!'


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO
)
logger = logging.getLogger(__name__)


if os.path.exists(MODEL_FOLDER):
    print(f'Found existing models folder at {MODEL_FOLDER}, skip creating a new one!')
    os.makedirs(MODEL_FOLDER, exist_ok=True)
else:
    os.makedirs(MODEL_FOLDER)

#########################################################################
# Download Model
#########################################################################
logger.info('Downloading models...')
download_model = partial(download_model_folder, DATA_FOLDER=MODEL_FOLDER)

# model size:  could be one of 'small' (GPT2 with 117M), 'medium'(345M) or 'large' (1542M)
# dataset: one of 'multiref' or 'dstc'
# from_scratch: True : load model trained from scratch or False: load model trained from fine-tuning the GPT-2
target_folder = download_model(model_size='medium', dataset='multiref', from_scratch=False)
logger.info('Done!\n')