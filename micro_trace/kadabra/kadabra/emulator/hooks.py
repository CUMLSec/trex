import os

from kadabra.utils.utils import addr_to_int, int_to_hex, to_unsinged
from unicorn import *
from kadabra.arch.arch_const import *

HOOK_MEM_RW = 0
HOOK_MEM_UNMAPPED = 1
HOOK_BASIC_BLOCK = 2
HOOK_INSTRUCTION = 3


def hook_mem_access(uc, access, address, size, value, emu):
    current_address = emu.reg_read(emu.arch.IP)
    value = to_unsinged(value, size * 8)

    if emu.stop_next_instruction:
        opcode = str(emu.mem_read(current_address, size)).encode("utf-8").hex()
        skip = False

        for op in emu.arch.returns:
            if opcode.startswith(op):
                if not emu.skip_return:
                    emu.mem_write(address, "\xdd\xdd\xdd\xdd")
                skip = True
        for op in emu.arch.calls:
            if opcode.startswith(op):
                emu.stop_execution()
                skip = True
        if not skip:
            emu.stop_execution()

    if access == UC_MEM_WRITE:

        if emu.verbosity_level > 1:
            print("Instruction 0x{:x} writes value 0x{:x} with 0x{:x} bytes into 0x{:x}".format(current_address, value,
                                                                                                size, address))
        value_hex = int_to_hex(value, size)
        prev_value = addr_to_int(emu.mem_read(address, size)) if address in emu.memory else 0
        emu.add_to_emulator_mem(address, value_hex)

    else:
        # memory read index replacement
        if emu.mem_read_index_map:
            if emu.mem_read_index_counter in emu.mem_read_index_map:
                value = emu.mem_read_index_map[emu.mem_read_index_counter]
                emu.mem_write(address, value)
            emu.mem_read_index_counter += 1

            if emu.stop_next_instruction:
                opcode = str(emu.mem_read(current_address, size)).encode("hex")
                if opcode.startswith("c3") or opcode.startswith("cb"):
                    value = addr_to_int("\xdd\xdd\xdd\xdd")
                    emu.mem_write(address, "\xdd\xdd\xdd\xdd")
                    SP = emu.reg_read(emu.arch.SP)
                    emu.reg_write(emu.arch.SP, SP + 4)
                if opcode.startswith("c2") or opcode.startswith("ca"):
                    value = addr_to_int("\xdd\xdd\xdd\xdd")
                    emu.mem_write(address, "\xdd\xdd\xdd\xdd")
                    SP = emu.reg_read(emu.arch.SP)
                    v = addr_to_int(opcode[1:])
                    emu.reg_write(emu.arch.SP, SP + 4 + v)

                if opcode.startswith("e8") or opcode.startswith("9a") or opcode.startswith("ff"):
                    emu.stop_execution()

        value = addr_to_int(emu.mem_read(address, size))

        if emu.no_zero_mem and value == 0:
            value = 1
            emu.mem_write(address, "\x01")

        prev_value = value

        if emu.verbosity_level > 1:
            print("Instruction 0x{:x} reads value 0x{:x} with 0x{:x} bytes from 0x{:x}".format(current_address, value,
                                                                                               size, address))
    if emu.memory_trace:
        emu.memory_tracer.add_trace(current_address, access, address, prev_value, value, size)

    return True


def get_inst(emu, address, size, last_registers=None):
    inst = bytes(emu.mem_read(address, size))

    static = ''
    for _, _, op_code, op_str in emu.md.disasm_lite(inst, address):
        # print(f'address:{address:x}, size:{size:x}, instruction:{op_code} {op_str}', end=' ')
        # static += f'{address:x}\tsize:{size:x}\t{op_code} {op_str}'
        static += f'{op_code} {op_str}'

    # prepare traces
    return static, emu.dump_registers()


def hook_mem_invalid(uc, access, address, size, value, emulator):
    value = to_unsinged(value, size * 8)

    if access == UC_MEM_WRITE_UNMAPPED or access == UC_MEM_READ_UNMAPPED:
        emulator.mem_map(address, size)
        # print('unmapped memory operation, allocating the memory on demand')
        if access == UC_MEM_READ_UNMAPPED:
            # print('create random memory value to read')
            emulator.mem_write(address, os.urandom(size))
        return True
    else:
        return False


def hook_code(uc, address, size, emu):
    opcode = emu.mem_read(address, size)

    static, registers = get_inst(emu, address, size)
    # if 'sw' in static:
    print(address, static, opcode)
    print(registers)
    print()

    emu.static.append(static)
    emu.trace.append(registers)

    for op in emu.arch.jumps.union(emu.arch.conditional_jumps).union(emu.arch.calls):
        if emu.arch.arch_id == ARCH_X86_64 or emu.arch.arch_id == ARCH_X86_32:
            if opcode.hex().startswith(op):
                emu.reg_write(emu.arch.IP, address + size)
                break
        else:
            try:
                if static.split()[0] == op:
                    emu.reg_write(emu.arch.IP, address + size)
                    break
            except IndexError:
                assert static == ""
                emu.reg_write(emu.arch.IP, address + size)
                break

    for op in emu.arch.returns:
        if emu.arch.arch_id == ARCH_X86_64 or emu.arch.arch_id == ARCH_X86_32:
            if opcode.hex().startswith(op):
                emu.stop_next_instruction = True
                emu.stop_execution()
                return False
        else:
            try:
                if static.split()[0] == op or static == op:
                    # print(static, opcode)
                    # print(registers)
                    # print()
                    emu.stop_next_instruction = True
                    emu.stop_execution()
                    return False
            except IndexError:
                assert static == ""
                emu.stop_next_instruction = True
                emu.stop_execution()
                return False

    if emu.stop_next_instruction:
        emu.stop_execution()

    if address == emu.final_instruction:
        emu.stop_next_instruction = True

        for op in emu.arch.jumps.union(emu.arch.conditional_jumps):
            if emu.arch.arch_id == ARCH_X86_64 or emu.arch.arch_id == ARCH_X86_32:
                if opcode.hex().startswith(op):
                    emu.stop_next_instruction = True
                    emu.stop_execution()
                    return False
            else:
                try:
                    if static.split()[0] == op or static == op:
                        # print(static, opcode)
                        # print(registers)
                        # print()
                        emu.stop_next_instruction = True
                        emu.stop_execution()
                        return False
                except IndexError:
                    assert static == ""
                    emu.stop_next_instruction = True
                    emu.stop_execution()
                    return False

    if emu.verbosity_level > 1:
        print("0x{:x};{}".format(address, opcode.hex()))

    return True


def hook_block(uc, address, size, emu):
    opcodes = str(emu.mem_read(address, size))

    # handle breakpoint
    if emu.basic_block_breakpoints_enabled and address in emu.basic_block_breakpoints:
        cb = emu.basic_block_breakpoints[address][0]
        args = emu.basic_block_breakpoints[address][1]
        call = cb(emu, *args)

        # bp handler returns False
        if not call:
            emu.stop_execution()

    if emu.verbosity_level > 1:
        print("Basic block at 0x{:x}".format(address))

    if emu.basic_block_trace:
        emu.code_tracer.add_basic_block_trace(address, opcodes, size)
    return True
