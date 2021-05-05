import subprocess
from multiprocessing import Pool

from command.configs import fields


def run(field):
    subprocess.run(
        ['fairseq-preprocess', '--only-source', '--srcdict', f'data-bin/pretrain/{field}/dict.txt', '--trainpref',
         f'data-src/data_structure/x64-O3/train.{field}',
         '--validpref',
         f'data-src/data_structure/x64-O3/valid.{field}', '--destdir', f'data-bin/data_structure/x64-O3/{field}',
         '--workers',
         '4'])


with Pool() as pool:
    pool.map(run, fields)

subprocess.run(
    ['fairseq-preprocess', '--only-source', '--srcdict', f'data-bin/label_dict.txt', '--trainpref',
     f'data-src/data_structure/x64-O3/train.label',
     '--validpref',
     f'data-src/data_structure/x64-O3/valid.label', '--destdir', f'data-bin/data_structure/x64-O3/label',
     '--workers', '16'])
