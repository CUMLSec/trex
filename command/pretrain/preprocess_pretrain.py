from command.configs import fields
from multiprocessing import Pool
import subprocess


def run(field):
    subprocess.run(
        ['fairseq-preprocess', '--only-source', '--srcdict', f'data-bin/pretrain_dict/{field}/dict.txt', '--trainpref',
         f'data-src/pretrain/train.{field}',
         '--validpref',
         f'data-src/pretrain/valid.{field}', '--destdir', f'data-bin/pretrain/{field}', '--workers',
         '40'])


with Pool() as pool:
    pool.map(run, fields)
