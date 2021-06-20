from kadabra.arch.arch_const import ARCH_X86_64, ARCH_X86_32, ARCH_ARM_32, ARCH_MIPS_32
from kadabra.emulator.emulator import Emulator


class EX:
    def __init__(self, code, mode='x86_64', md=None):

        if mode == 'x86_64':
            mode_name = ARCH_X86_64
        elif mode == 'x86':
            mode_name = ARCH_X86_32
        elif mode == 'arm':
            mode_name = ARCH_ARM_32
        elif mode == 'mips':
            mode_name = ARCH_MIPS_32
        else:
            raise NotImplementedError()

        self.mu = Emulator(mode_name, md)

        self.mu.initialise_regs_random()
        if mode == 'arm':
            self.mu.reg_write('APSR', 0xFFFFFFFF)

        self._CODE_BASE = 0x10000
        self._CODE_SIZE = 2 * 1024 * 1024

        self._STACK_BASE = self._CODE_BASE + 2 * self._CODE_SIZE
        self._STACK_SIZE = 4 * 1024 * 1024

        self.code = code
        self.mu.mem_map(self._CODE_BASE, self._CODE_SIZE)
        self.mu.mem_map(self._STACK_BASE, self._STACK_SIZE)

        self.mu.mem_write(self._CODE_BASE, code)

        # stack pointer in the middle
        self.mu.reg_write(self.mu.arch.SP, self._STACK_BASE + int(self._STACK_SIZE / 2))
        self.mu.reg_write(self.mu.arch.SB, self._STACK_BASE + int(self._STACK_SIZE / 4))

        # self.mu.hook_add(UC_HOOK_CODE, hook_code, user_data=md, begin=1, end=0, arg1=0)
        self.mu.set_traces(instruction=True)
        self.mu.set_hooks(mem_unmapped=True)

    def run(self, timeout=60000):
        self.mu.start_execution(self._CODE_BASE, self._CODE_BASE + len(self.code), timeout=timeout)
