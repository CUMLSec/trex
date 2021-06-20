from collections import OrderedDict
from unicorn import UC_ARCH_ARM64, UC_MODE_ARM

from unicorn.arm64_const import *


class ARM_64:
    def __init__(self):
        self.SB = "FP"
        self.SP = "SP"
        self.IP = "PC"
        self.FLAGS = "CPSR"
        self.segment_registers = set(["SB","SL","IP","LR","PC"])

        self.uc_arch = UC_ARCH_ARM64
        self.uc_mode = UC_MODE_ARM

        self.size = 64

        self.registers = OrderedDict([("X0", (UC_ARM64_REG_X0, 64)),
                                      ("X1", (UC_ARM64_REG_X1, 64)),
                                      ("X2", (UC_ARM64_REG_X2, 64)),
                                      ("X3", (UC_ARM64_REG_X3, 64)),
                                      ("X4", (UC_ARM64_REG_X4, 64)),
                                      ("X5", (UC_ARM64_REG_X5, 64)),
                                      ("X6", (UC_ARM64_REG_X6, 64)),
                                      ("X7", (UC_ARM64_REG_X7, 64)),
                                      ("X8", (UC_ARM64_REG_X8, 64)),
                                      ("X9", (UC_ARM64_REG_X9, 64)),
                                      ("X10", (UC_ARM64_REG_X10, 64)),
                                      ("X11", (UC_ARM64_REG_X11, 64)),
                                      ("X12", (UC_ARM64_REG_X12, 64)),
                                      ("X13", (UC_ARM64_REG_X13, 64)),
                                      ("X14", (UC_ARM64_REG_X14, 64)),
                                      ("X15", (UC_ARM64_REG_X15, 64)),
                                      ("X16", (UC_ARM64_REG_X16, 64)),
                                      ("X17", (UC_ARM64_REG_X17, 64)),
                                      ("X18", (UC_ARM64_REG_X18, 64)),
                                      ("X19", (UC_ARM64_REG_X19, 64)),
                                      ("X20", (UC_ARM64_REG_X20, 64)),
                                      ("X21", (UC_ARM64_REG_X21, 64)),
                                      ("X22", (UC_ARM64_REG_X22, 64)),
                                      ("X23", (UC_ARM64_REG_X23, 64)),
                                      ("X24", (UC_ARM64_REG_X24, 64)),
                                      ("X25", (UC_ARM64_REG_X25, 64)),
                                      ("X26", (UC_ARM64_REG_X26, 64)),
                                      ("X27", (UC_ARM64_REG_X27, 64)),
                                      ("X28", (UC_ARM64_REG_X28, 64)),
                                      ("X29", (UC_ARM64_REG_X29, 64)),
                                      ("X30", (UC_ARM64_REG_X30, 64)),

                                      ("W0", (UC_ARM64_REG_W0, 32)),
                                      ("W1", (UC_ARM64_REG_W1, 32)),
                                      ("W2", (UC_ARM64_REG_W2, 32)),
                                      ("W3", (UC_ARM64_REG_W3, 32)),
                                      ("W4", (UC_ARM64_REG_W4, 32)),
                                      ("W5", (UC_ARM64_REG_W5, 32)),
                                      ("W6", (UC_ARM64_REG_W6, 32)),
                                      ("W7", (UC_ARM64_REG_W7, 32)),
                                      ("W8", (UC_ARM64_REG_W8, 32)),
                                      ("W9", (UC_ARM64_REG_W9, 32)),
                                      ("W10", (UC_ARM64_REG_W10, 32)),
                                      ("W11", (UC_ARM64_REG_W11, 32)),
                                      ("W12", (UC_ARM64_REG_W12, 32)),
                                      ("W13", (UC_ARM64_REG_W13, 32)),
                                      ("W14", (UC_ARM64_REG_W14, 32)),
                                      ("W15", (UC_ARM64_REG_W15, 32)),
                                      ("W16", (UC_ARM64_REG_W16, 32)),
                                      ("W17", (UC_ARM64_REG_W17, 32)),
                                      ("W18", (UC_ARM64_REG_W18, 32)),
                                      ("W19", (UC_ARM64_REG_W19, 32)),
                                      ("W20", (UC_ARM64_REG_W20, 32)),
                                      ("W21", (UC_ARM64_REG_W21, 32)),
                                      ("W22", (UC_ARM64_REG_W22, 32)),
                                      ("W23", (UC_ARM64_REG_W23, 32)),
                                      ("W24", (UC_ARM64_REG_W24, 32)),
                                      ("W25", (UC_ARM64_REG_W25, 32)),
                                      ("W26", (UC_ARM64_REG_W26, 32)),
                                      ("W27", (UC_ARM64_REG_W27, 32)),
                                      ("W28", (UC_ARM64_REG_W28, 32)),
                                      ("W29", (UC_ARM64_REG_W29, 32)),
                                      ("W30", (UC_ARM64_REG_W30, 32)),

                                      ("D0", (UC_ARM64_REG_D0, 32)),
                                      ("D1", (UC_ARM64_REG_D1, 32)),
                                      ("D2", (UC_ARM64_REG_D2, 32)),
                                      ("D3", (UC_ARM64_REG_D3, 32)),
                                      ("D4", (UC_ARM64_REG_D4, 32)),
                                      ("D5", (UC_ARM64_REG_D5, 32)),
                                      ("D6", (UC_ARM64_REG_D6, 32)),
                                      ("D7", (UC_ARM64_REG_D7, 32)),
                                      ("D8", (UC_ARM64_REG_D8, 32)),
                                      ("D9", (UC_ARM64_REG_D9, 32)),
                                      ("D10", (UC_ARM64_REG_D10, 32)),
                                      ("D11", (UC_ARM64_REG_D11, 32)),
                                      ("D12", (UC_ARM64_REG_D12, 32)),
                                      ("D13", (UC_ARM64_REG_D13, 32)),
                                      ("D14", (UC_ARM64_REG_D14, 32)),
                                      ("D15", (UC_ARM64_REG_D15, 32)),
                                      ("D16", (UC_ARM64_REG_D16, 32)),
                                      ("D17", (UC_ARM64_REG_D17, 32)),
                                      ("D18", (UC_ARM64_REG_D18, 32)),
                                      ("D19", (UC_ARM64_REG_D19, 32)),
                                      ("D20", (UC_ARM64_REG_D20, 32)),
                                      ("D21", (UC_ARM64_REG_D21, 32)),
                                      ("D22", (UC_ARM64_REG_D22, 32)),
                                      ("D23", (UC_ARM64_REG_D23, 32)),
                                      ("D24", (UC_ARM64_REG_D24, 32)),
                                      ("D25", (UC_ARM64_REG_D25, 32)),
                                      ("D26", (UC_ARM64_REG_D26, 32)),
                                      ("D27", (UC_ARM64_REG_D27, 32)),
                                      ("D28", (UC_ARM64_REG_D28, 32)),
                                      ("D29", (UC_ARM64_REG_D29, 32)),
                                      ("D30", (UC_ARM64_REG_D30, 32)),

                                      ])

        self.conditional_jumps = set([
        ])
        self.jumps = set(["b",
                          "bx",
                          "bxj",
                          "jz"
                          ])

        self.returns = set(["ret",
                            "reti",
                            "c2",
                            "ca"])

        self.calls = set(["bl",
                          "blx"])
