import glob
import json
import re
import subprocess
from itertools import product

from capstone import *
import random
import string


def objdump(file, arch='x86', mode='32'):
    try:
        if arch == 'x86':
            result = subprocess.run(['objdump', '-d', file, '-j', '.text'], capture_output=True, text=True, check=True)
        elif arch == 'arm':
            if mode == '32':
                result = subprocess.run(['arm-linux-gnueabi-objdump', '-d', file, '-j', '.text'], capture_output=True,
                                        text=True, check=True)
            elif mode == '64':
                result = subprocess.run(['aarch64-linux-gnu-objdump', '-d', file, '-j', '.text'], capture_output=True,
                                        text=True, check=True)
            else:
                print(f'invalid size, has to be either 32 or 64')

        elif arch == 'mips':
            if mode == '32':
                result = subprocess.run(['mips-linux-gnu-objdump', '-d', file, '-j', '.text'], capture_output=True,
                                        text=True, check=True)
            elif mode == '64':
                result = subprocess.run(['mips64-linux-gnuabi64-objdump', '-d', file, '-j', '.text'],
                                        capture_output=True, text=True, check=True)
            else:
                print(f'invalid size, has to be either 32 or 64')

    except subprocess.CalledProcessError:
        print(f'cannot dump the file {result.stderr}')

    return result.stdout


# input: the .text part of the objdump's output for one elf file
# output:the dictionary, function_name:bytes delimited by space
def parse_objdump_output(s, arch='x86'):
    funcname = ''
    ret_func_dict = {}

    for line in s.split('\n'):
        # function name, and this is the start
        funcname_re = re.findall(r'\<(.*)\>:', line)
        if len(funcname_re) == 1:
            funcname = funcname_re[0]
            ret_func_dict[funcname] = []
            continue

        elif len(funcname_re) > 1:
            print(f'error, more than one functions matched: {funcname_re}')
            exit()

        # match bytes
        bytes_re = re.findall(r':\t(.*?)\t', line)
        if len(bytes_re) == 1:
            inst_bytes = bytes_re[0].strip()
            if arch == 'x86':
                ret_func_dict[funcname].extend(inst_bytes.split())
            elif arch == 'arm':
                assert len(inst_bytes) == 8, print(f'arm instruction {inst_bytes} length (byte) must be 8')
                ret_func_dict[funcname].extend([inst_bytes[b:b + 2] for b in range(0, len(inst_bytes), 2)][::-1])
            elif arch == 'mips':
                assert len(inst_bytes) == 8, print(f'mips instruction {inst_bytes} length (byte) must be 8')
                ret_func_dict[funcname].extend([inst_bytes[b:b + 2] for b in range(0, len(inst_bytes), 2)])
            else:
                print(f'error, unknown architecture: {arch}')
                exit()

        elif len(bytes_re) == 0:
            # handle nop line
            bytes_re_nop = re.findall(r':\t(.*?)\s+$', line)
            if len(bytes_re_nop) == 1:
                inst_bytes = bytes_re_nop[0].strip()
                if arch == 'x86':
                    ret_func_dict[funcname].extend(inst_bytes.split())
                elif arch == 'arm':
                    assert len(inst_bytes) == 8, print(f'arm instruction {inst_bytes} length (byte) must be 8')
                    ret_func_dict[funcname].extend([inst_bytes[b:b + 2] for b in range(0, len(inst_bytes), 2)][::-1])
                elif arch == 'mips':
                    assert len(inst_bytes) == 8, print(f'mips instruction {inst_bytes} length (byte) must be 8')
                    ret_func_dict[funcname].extend([inst_bytes[b:b + 2] for b in range(0, len(inst_bytes), 2)])
                else:
                    print(f'error, unknown architecture: {arch}')
                    exit()

        elif len(bytes_re) > 1:
            print(f'error, more than one byte strings matched: {bytes_re}')
            exit()

    return ret_func_dict


# archs = ['x86', 'arm', 'mips']
# modes = ['32', '64']
# opts = ['O0', 'O1', 'O2', 'O3', 'orig',
#         'bcfobf', 'cffobf', 'splitobf', 'subobf', 'acdobf', 'indibran', 'strcry', 'funcwra']

# archs = ['x86', 'arm', 'mips']
# modes = ['32', '64']
# opts = ['O0', 'O1', 'O2', 'O3', 'orig',
#         'bcfobf', 'cffobf', 'splitobf', 'subobf', 'indibran']
# opts = ['O0', 'O1', 'O2', 'O3']

archs = ['x86']
modes = ['64']
# opts = ['O0', 'O1', 'O2', 'O3', 'orig',
#         'bcfobf', 'cffobf', 'splitobf', 'subobf', 'indibran']
opts = ['O0', 'O1', 'O2', 'O3']

mds = {}
mds['x86-32'] = Cs(CS_ARCH_X86, CS_MODE_32)
mds['x86-64'] = Cs(CS_ARCH_X86, CS_MODE_64)
mds['arm-32'] = Cs(CS_ARCH_ARM, CS_MODE_ARM + CS_MODE_LITTLE_ENDIAN)
mds['arm-64'] = Cs(CS_ARCH_ARM64, CS_MODE_ARM + CS_MODE_LITTLE_ENDIAN)
mds['mips-32'] = Cs(CS_ARCH_MIPS, CS_MODE_MIPS32 + CS_MODE_LITTLE_ENDIAN)
mds['mips-64'] = Cs(CS_ARCH_MIPS, CS_MODE_MIPS64 + CS_MODE_LITTLE_ENDIAN)

for arch, mode, opt in product(archs, modes, opts):
    # skip arm-64 and mips-64
    if (arch == 'arm' or arch == 'mips') and mode == '64':
        continue

    md = mds[f'{arch}-{mode}']
    file_list = glob.glob(f'data-raw/bin/{arch}-{mode}/*-{opt}/*', recursive=True)
    func_dict = {}

    for i, file in enumerate(file_list):
        objdump_result = objdump(file, arch=arch, mode=mode)
        ret_func_dict = parse_objdump_output(objdump_result, arch=arch)
        for funcname, funcbody in ret_func_dict.items():
            if funcname in func_dict:
                func_dict[funcname][f'{file}'] = funcbody
            else:
                func_dict[funcname] = {f'{file}': funcbody}

    if func_dict:
        with open(f'data-raw/funcbytes/{arch}-{mode}-{opt}', 'w') as f:
            json.dump(func_dict, f)
