from kadabra.arch.x86.x86_64 import X86_64
from kadabra.arch.x86.x86_32 import X86_32
from kadabra.arch.mips.mips_32 import MIPS_32
from kadabra.arch.arm.arm_32 import ARM_32
from kadabra.arch.arch_const import *


class Architecture:
    def __init__(self, arch_id):

        self.arch_id = arch_id
        if arch_id == ARCH_X86_64:
            arch = X86_64()
        elif arch_id == ARCH_X86_32:
            arch = X86_32()
        elif arch_id == ARCH_ARM_32:
            arch = ARM_32()
        elif arch_id == ARCH_MIPS_32:
            arch = MIPS_32()
        else:
            raise NotImplementedError()

        self.IP = arch.IP
        self.SP = arch.SP
        self.SB = arch.SB
        self.FLAGS = arch.FLAGS
        self.segment_registers = arch.segment_registers

        self.conditional_jumps = arch.conditional_jumps
        self.jumps = arch.jumps
        self.returns = arch.returns
        self.calls = arch.calls

        self.uc_mode = arch.uc_mode
        self.uc_arch = arch.uc_arch

        self.registers = arch.registers
        self.size = arch.size
