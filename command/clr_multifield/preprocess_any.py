from command.configs import fields
from multiprocessing import Pool
import subprocess
import os
import shutil


def run(field):
    subprocess.run(
        ['fairseq-preprocess', '--only-source', '--srcdict', f'data-bin/pretrain/{field}/dict.txt', '--trainpref',
         f'data-src/clr_multifield_any/train.{field}.input0',
         '--validpref',
         f'data-src/clr_multifield_any/valid.{field}.input0', '--destdir',
         f'data-bin/clr_multifield_any/input0/{field}',
         '--workers',
         '40'])
    subprocess.run(
        ['fairseq-preprocess', '--only-source', '--srcdict', f'data-bin/pretrain/{field}/dict.txt', '--trainpref',
         f'data-src/clr_multifield_any/train.{field}.input1',
         '--validpref',
         f'data-src/clr_multifield_any/valid.{field}.input1', '--destdir',
         f'data-bin/clr_multifield_any/input1/{field}',
         '--workers',
         '40'])


with Pool() as pool:
    pool.map(run, fields)

directory = 'data-bin/clr_multifield_any/label/'
if not os.path.exists(directory):
    os.mkdir(directory)

shutil.copy('data-src/clr_multifield_any/train.label', directory)
shutil.copy('data-src/clr_multifield_any/valid.label', directory)
