from command.configs import fields
from multiprocessing import Pool
import subprocess
import os
import shutil


def run(field):
    subprocess.run(
        ['fairseq-preprocess', '--only-source', '--srcdict', f'data-bin/data_noisy_mmod/{field}/dict.txt', '--trainpref',
         f'data-src/binmatch_clr_multifield/train.{field}.input0',
         '--validpref',
         f'data-src/binmatch_clr_multifield/valid.{field}.input0', '--destdir',
         f'data-bin/binmatch_clr_multifield/input0/{field}',
         '--workers',
         '40'])
    subprocess.run(
        ['fairseq-preprocess', '--only-source', '--srcdict', f'data-bin/data_noisy_mmod/{field}/dict.txt',
         '--trainpref',
         f'data-src/binmatch_clr_multifield/train.{field}.input1',
         '--validpref',
         f'data-src/binmatch_clr_multifield/valid.{field}.input1', '--destdir',
         f'data-bin/binmatch_clr_multifield/input1/{field}',
         '--workers',
         '40'])


with Pool() as pool:
    pool.map(run, fields)

directory = 'data-bin/binmatch_clr_multifield/label/'
if not os.path.exists(directory):
    os.mkdir(directory)

shutil.copy('data-src/binmatch_clr_multifield/train.label', directory)
shutil.copy('data-src/binmatch_clr_multifield/valid.label', directory)
