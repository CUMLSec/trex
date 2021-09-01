from command.configs import fields
from multiprocessing import Pool
import subprocess
import os
import shutil


def run(field):
    subprocess.run(
        ['fairseq-preprocess', '--only-source', '--srcdict', f'data-bin/pretrain_dict/{field}/dict.txt', '--trainpref',
         f'data-src/similarity_x64/train.{field}.input0',
         '--validpref',
         f'data-src/similarity_x64/valid.{field}.input0', '--destdir',
         f'data-bin/similarity_x64/input0/{field}',
         '--workers',
         '40'])
    subprocess.run(
        ['fairseq-preprocess', '--only-source', '--srcdict', f'data-bin/pretrain_dict/{field}/dict.txt', '--trainpref',
         f'data-src/similarity_x64/train.{field}.input1',
         '--validpref',
         f'data-src/similarity_x64/valid.{field}.input1', '--destdir',
         f'data-bin/similarity_x64/input1/{field}',
         '--workers',
         '40'])


with Pool() as pool:
    pool.map(run, fields)

directory = 'data-bin/similarity_x64/label/'
if not os.path.exists(directory):
    os.mkdir(directory)

shutil.copy('data-src/similarity_x64/train.label', directory)
shutil.copy('data-src/similarity_x64/valid.label', directory)
