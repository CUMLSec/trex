import glob
import re
import subprocess
from collections import defaultdict
from itertools import product

from capstone import *
import json

from tracing import EX
from unicorn.unicorn import UcError

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
    num = s.replace('0x', '')
    assert len(num) <= 8
    num = '0' * (8 - len(num)) + num
    return num


def get_trace(static_codes, registers_list):
    code = []
    trace = []
    inst_indices = []
    token_indices = []
    for inst_index, (inst, registers) in enumerate(zip(static_codes, registers_list)):
        tokens = tokenize(inst)

        for token_index, token in enumerate(tokens):
            if token.upper() in registers:
                code.append(token.lower())
                trace.append(hex2str(hex(registers[token.upper()])))
            elif '0x' in token.lower():
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


modes = ['arm', 'mips', 'x86', 'x86_64']
# modes = ['mips']

opts = ['O0', 'O1', 'O2', 'O3',
        'bcfobf', 'cffobf', 'splitobf', 'subobf', 'acdobf', 'indibran', 'strcry', 'funcwra']
# opts = ['O0', 'O1', 'O2', 'O3']

mds = {'x86-32': Cs(CS_ARCH_X86, CS_MODE_32),
       'x86-64': Cs(CS_ARCH_X86, CS_MODE_64),
       'arm-32': Cs(CS_ARCH_ARM, CS_MODE_ARM + CS_MODE_LITTLE_ENDIAN),
       'arm-64': Cs(CS_ARCH_ARM64, CS_MODE_ARM + CS_MODE_LITTLE_ENDIAN),
       'mips-32': Cs(CS_ARCH_MIPS, CS_MODE_MIPS32 + CS_MODE_BIG_ENDIAN),
       'mips-64': Cs(CS_ARCH_MIPS, CS_MODE_MIPS64 + CS_MODE_BIG_ENDIAN)}

func_dict = {}
num_traces = 100
timeout = 500000

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
            func_dict = json.load(f)

        with open(f'data-raw/functraces/{arch}-{bit}-{opt}', 'w') as wf:
            functraces = {}
            for i, funcname in enumerate(func_dict):
                if funcname.startswith('_'):
                    continue

                # if funcname != 'main':
                #     continue

                for j, binfile in enumerate(func_dict[funcname]):

                    # if binfile != 'data-raw/bin/arm-32/coreutils-8.26-O0/sha384sum':
                    #     continue

                    # funcbody is list of bytes, we covert bytes to instructions
                    code = str2byte(' '.join(func_dict[funcname][binfile]))

                    # skip functions that are too short
                    if num_inst(code, md) < 5:
                        continue

                    print(binfile, '...')
                    for _ in range(num_traces):
                        # create emulator for dynamic tracing micro execution
                        ex = EX(code, mode=mode, md=md)

                        try:
                            ex.run(timeout=timeout)
                        except Exception as e:
                            print(f"{e}: {binfile} {funcname}")

                        instruction_body, traces, inst_indices, token_indices = get_trace(ex.mu.static, ex.mu.trace)

                        if funcname in functraces and binfile in functraces[funcname]:
                            functraces[funcname][f'{binfile}'].append(
                                (instruction_body, traces, inst_indices, token_indices))
                        elif funcname in functraces:
                            functraces[funcname][f'{binfile}'] = [
                                (instruction_body, traces, inst_indices, token_indices)]
                        else:
                            functraces[funcname] = {
                                f'{binfile}': [(instruction_body, traces, inst_indices, token_indices)]}

                    # dummy trace to enforce model to focus on static prediction
                    traces_dummy = ['#' * 8] * len(traces)
                    functraces[funcname][f'{binfile}'].append(
                        (instruction_body, traces_dummy, inst_indices, token_indices))

                    # tokens = byte2instruction(code, md)
                    # for token in tokens.split():
                    #     token_dict[token] += 1

                exit()
            json.dump(functraces, wf)

    # with open(f'data-raw/token_dict/{arch}-{mode}', 'w') as f:
    #     json.dump(token_dict, f)

# with open('data-raw/funcpair/func_dict', 'w') as f:
#     json.dump(func_dict, f)
