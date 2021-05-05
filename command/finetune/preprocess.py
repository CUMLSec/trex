from command.configs import fields
from multiprocessing import Pool
import subprocess
import os
import shutil


def run(field):
    subprocess.run(
        ['fairseq-preprocess', '--only-source', '--srcdict', f'data-bin/pretrain_dict/{field}/dict.txt', '--trainpref',
         f'data-src/similarity/train.{field}.input0',
         '--validpref',
         f'data-src/similarity/valid.{field}.input0', '--destdir',
         f'data-bin/similarity/input0/{field}',
         '--workers',
         '40'])
    subprocess.run(
        ['fairseq-preprocess', '--only-source', '--srcdict', f'data-bin/pretrain_dict/{field}/dict.txt', '--trainpref',
         f'data-src/similarity/train.{field}.input1',
         '--validpref',
         f'data-src/similarity/valid.{field}.input1', '--destdir',
         f'data-bin/similarity/input1/{field}',
         '--workers',
         '40'])


with Pool() as pool:
    pool.map(run, fields)

directory = 'data-bin/similarity/label/'
if not os.path.exists(directory):
    os.mkdir(directory)

shutil.copy('data-src/similarity/train.label', directory)
shutil.copy('data-src/similarity/valid.label', directory)
