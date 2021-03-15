from command.configs import fields, byte_start_pos
from multiprocessing import Pool
import subprocess


def run(field):
    subprocess.run(
        ['fairseq-preprocess', '--only-source', '--trainpref', f'data-src/pretrain_10k/train.{field}',
         '--validpref',
         f'data-src/pretrain_10k/valid.{field}', '--destdir', f'data-bin/pretrain_10k/{field}', '--workers',
         '40'])


def run_byte_with_dict(field):
    subprocess.run(
        ['fairseq-preprocess', '--only-source', '--srcdict', 'data-bin/byte_dict.txt', '--trainpref',
         f'data-src/pretrain_10k/train.{field}',
         '--validpref',
         f'data-src/pretrain_10k/valid.{field}', '--destdir', f'data-bin/pretrain_10k/{field}',
         '--workers',
         '40'])


with Pool() as pool:
    pool.map(run_byte_with_dict, fields[byte_start_pos:])

with Pool() as pool:
    pool.map(run, fields[:byte_start_pos])
