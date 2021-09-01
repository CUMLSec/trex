import glob
import re
import subprocess
from collections import defaultdict
from itertools import product

from capstone import *
import json

import os.path


def str2byte(s):
    return bytes(bytearray.fromhex(s))


def tokenize(s):
    s = s.replace(',', ' , ')
    s = s.replace('[', ' [ ')
    s = s.replace(']', ' ] ')
    s = s.replace(':', ' : ')
    s = s.replace('*', ' * ')
    s = s.replace('(', ' ( ')
    s = s.replace(')', ' ) ')
    s = s.replace('{', ' { ')
    s = s.replace('}', ' } ')
    s = s.replace('#', '')
    s = s.replace('$', '')
    s = s.replace('!', ' ! ')

    s = re.sub(r'-(0[xX][0-9a-fA-F]+)', r'- \1', s)
    s = re.sub(r'-([0-9a-fA-F]+)', r'- \1', s)

    # s = re.sub(r'0[xX][0-9a-fA-F]+', 'hexvar', s)

    return s.split()


def byte2instruction(s, md):
    ret_tokens = []

    try:
        for _, _, op_code, op_str in md.disasm_lite(s, 0x400000):
            tokens = tokenize(f'{op_code} {op_str}')
            for token in tokens:
                ret_tokens.append(token)
    except CsError as e:
        print("ERROR: %s" % e)

    return ' '.join(ret_tokens)


def num_inst(s, md):
    return len(list(md.disasm_lite(s, 0x400000)))


def hex2str(s):
    num = s.replace('0x', '')[-8:]
    assert len(num) <= 8
    num = '0' * (8 - len(num)) + num
    return num


def get_trace(static_codes, md):
    code = []
    trace = []
    inst_indices = []
    token_indices = []

    for inst_index, (_, _, op_code, op_str) in enumerate(md.disasm_lite(static_codes, 0x400000)):

        tokens = tokenize(f'{op_code} {op_str}')

        for token_index, token in enumerate(tokens):
            if '0x' in token.lower():
                code.append('hexvar')
                trace.append(hex2str(token.lower()))

            elif token.lower().isdigit():
                code.append('num')
                trace.append(hex2str(token.lower()))
            else:
                code.append(token.lower())
                trace.append('#' * 8)
            inst_indices.append(inst_index)
            token_indices.append(token_index)

    return code, trace, inst_indices, token_indices


# modes = ['arm', 'mips', 'x86', 'x86_64']
modes = ['x86_64']

# opts = ['O0', 'O1', 'O2', 'O3',
#         'bcfobf', 'cffobf', 'splitobf', 'subobf', 'acdobf', 'indibran', 'strcry', 'funcwra']
opts = ['O0', 'O1', 'O2', 'O3']

mds = {'x86-32': Cs(CS_ARCH_X86, CS_MODE_32),
       'x86-64': Cs(CS_ARCH_X86, CS_MODE_64),
       'arm-32': Cs(CS_ARCH_ARM, CS_MODE_ARM + CS_MODE_LITTLE_ENDIAN),
       'arm-64': Cs(CS_ARCH_ARM64, CS_MODE_ARM + CS_MODE_LITTLE_ENDIAN),
       'mips-32': Cs(CS_ARCH_MIPS, CS_MODE_MIPS32 + CS_MODE_BIG_ENDIAN),
       'mips-64': Cs(CS_ARCH_MIPS, CS_MODE_MIPS64 + CS_MODE_BIG_ENDIAN)}

func_dict = {}

for mode in modes:
    if mode == 'x86_64':
        md = mds['x86-64']
        arch = 'x86'
        bit = '64'
    elif mode == 'x86':
        md = mds['x86-32']
        arch = 'x86'
        bit = '32'
    elif mode == 'arm':
        md = mds['arm-32']
        arch = 'arm'
        bit = '32'
    elif mode == 'mips':
        md = mds['mips-32']
        arch = 'mips'
        bit = '32'
    else:
        raise NotImplementedError()

    for opt in opts:
        if not os.path.isfile(f'data-raw/funcbytes/{arch}-{bit}-{opt}'):
            continue

        with open(f'data-raw/funcbytes/{arch}-{bit}-{opt}', 'r') as f:
            print(f'json loading dictionary {arch}-{bit}-{opt}...')
            func_dict = json.load(f)

        # create directory if it doesn't exist'
        if not os.path.isdir(f'data-raw/functraces/{arch}-{bit}-{opt}'):
            os.mkdir(f'data-raw/functraces/{arch}-{bit}-{opt}')

        for i, funcname in enumerate(func_dict):
            if funcname.startswith('_'):
                continue

            # if funcname != 'main':
            #     continue

            if not os.path.isdir(f'data-raw/functraces/{arch}-{bit}-{opt}/{funcname}'):
                os.mkdir(f'data-raw/functraces/{arch}-{bit}-{opt}/{funcname}')

            for j, binfile in enumerate(func_dict[funcname]):

                # if binfile != 'data-raw/bin/arm-32/coreutils-8.26-O0/sha384sum':
                #     continue

                # funcbody is list of bytes, we covert bytes to instructions
                code = str2byte(' '.join(func_dict[funcname][binfile]))

                # skip functions that are too short
                if num_inst(code, md) < 5 or num_inst(code, md) > 512:
                    continue

                print(binfile, '...')

                proj = '-'.join(binfile.split('/')[-2].split('-')[:-1])
                filename = binfile.split('/')[-1]

                with open(f'data-raw/functraces/{arch}-{bit}-{opt}/{funcname}/{proj}-{filename}', 'w') as wf:
                    functraces = []
                    instruction_body, traces, inst_indices, token_indices = get_trace(code, md)
                    functraces.append((instruction_body, traces, inst_indices, token_indices))

                    # tokens = byte2instruction(code, md)
                    # for token in tokens.split():
                    #     token_dict[token] += 1

                    json.dump(functraces, wf)

    # with open(f'data-raw/token_dict/{arch}-{mode}', 'w') as f:
    #     json.dump(token_dict, f)

# with open('data-raw/funcpair/func_dict', 'w') as f:
#     json.dump(func_dict, f)
