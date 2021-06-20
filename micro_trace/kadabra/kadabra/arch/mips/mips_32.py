from collections import OrderedDict
from unicorn import UC_ARCH_MIPS, UC_MODE_MIPS32, UC_MODE_BIG_ENDIAN

from unicorn.mips_const import *


class MIPS_32:
    def __init__(self):
        self.SB = "FP"
        self.SP = "SP"
        self.IP = "PC"
        self.FLAGS = "ZERO"
        self.segment_registers = set(["AT", "GP", "K0", "K1", "PC"])

        self.uc_arch = UC_ARCH_MIPS
        self.uc_mode = UC_MODE_MIPS32 + UC_MODE_BIG_ENDIAN

        self.size = 32

        self.registers = OrderedDict([
            ("PC", (UC_MIPS_REG_PC, 32)),

            ("ZERO", (UC_MIPS_REG_ZERO, 32)),

            ("AT", (UC_MIPS_REG_AT, 32)),

            ("V0", (UC_MIPS_REG_V0, 32)),
            ("V1", (UC_MIPS_REG_V1, 32)),

            ("A0", (UC_MIPS_REG_A0, 32)),
            ("A1", (UC_MIPS_REG_A1, 32)),
            ("A2", (UC_MIPS_REG_A2, 32)),
            ("A3", (UC_MIPS_REG_A3, 32)),

            ("T0", (UC_MIPS_REG_T0, 32)),
            ("T1", (UC_MIPS_REG_T1, 32)),
            ("T2", (UC_MIPS_REG_T2, 32)),
            ("T3", (UC_MIPS_REG_T3, 32)),
            ("T4", (UC_MIPS_REG_T4, 32)),
            ("T5", (UC_MIPS_REG_T5, 32)),
            ("T6", (UC_MIPS_REG_T6, 32)),
            ("T7", (UC_MIPS_REG_T7, 32)),
            ("T8", (UC_MIPS_REG_T8, 32)),
            ("T9", (UC_MIPS_REG_T9, 32)),

            ("S0", (UC_MIPS_REG_S0, 32)),
            ("S1", (UC_MIPS_REG_S1, 32)),
            ("S2", (UC_MIPS_REG_S2, 32)),
            ("S3", (UC_MIPS_REG_S3, 32)),
            ("S4", (UC_MIPS_REG_S4, 32)),
            ("S5", (UC_MIPS_REG_S5, 32)),
            ("S6", (UC_MIPS_REG_S6, 32)),
            ("S7", (UC_MIPS_REG_S7, 32)),

            ("K0", (UC_MIPS_REG_K0, 32)),
            ("K1", (UC_MIPS_REG_K1, 32)),

            ("GP", (UC_MIPS_REG_GP, 32)),

            ("SP", (UC_MIPS_REG_SP, 32)),

            ("FP", (UC_MIPS_REG_FP, 32)),

            ("RA", (UC_MIPS_REG_RA, 32)),

            ("F0", (UC_MIPS_REG_F0, 32)),
            ("F1", (UC_MIPS_REG_F1, 32)),
            ("F2", (UC_MIPS_REG_F2, 32)),
            ("F3", (UC_MIPS_REG_F3, 32)),
            ("F4", (UC_MIPS_REG_F4, 32)),
            ("F5", (UC_MIPS_REG_F5, 32)),
            ("F6", (UC_MIPS_REG_F6, 32)),
            ("F7", (UC_MIPS_REG_F7, 32)),
            ("F8", (UC_MIPS_REG_F8, 32)),
            ("F9", (UC_MIPS_REG_F9, 32)),
            ("F10", (UC_MIPS_REG_F10, 32)),
            ("F11", (UC_MIPS_REG_F11, 32)),
            ("F12", (UC_MIPS_REG_F12, 32)),
            ("F13", (UC_MIPS_REG_F13, 32)),
            ("F14", (UC_MIPS_REG_F14, 32)),
            ("F15", (UC_MIPS_REG_F15, 32)),
            ("F16", (UC_MIPS_REG_F16, 32)),
            ("F17", (UC_MIPS_REG_F17, 32)),
            ("F18", (UC_MIPS_REG_F18, 32)),
            ("F19", (UC_MIPS_REG_F19, 32)),
            ("F20", (UC_MIPS_REG_F20, 32)),
            ("F21", (UC_MIPS_REG_F21, 32)),
            ("F22", (UC_MIPS_REG_F22, 32)),
            ("F23", (UC_MIPS_REG_F23, 32)),
            ("F24", (UC_MIPS_REG_F24, 32)),
            ("F25", (UC_MIPS_REG_F25, 32)),
            ("F26", (UC_MIPS_REG_F26, 32)),
            ("F27", (UC_MIPS_REG_F27, 32)),
            ("F28", (UC_MIPS_REG_F28, 32)),
            ("F29", (UC_MIPS_REG_F29, 32)),
            ("F30", (UC_MIPS_REG_F30, 32)),
            ("F31", (UC_MIPS_REG_F31, 32)),
        ])

        self.conditional_jumps = set(["bc1f",
                                      "bc1t",
                                      "beq",
                                      "beqz",
                                      "bne",
                                      "bnez",
                                      "bltz",
                                      "bgez",
                                      "blez",
                                      "bgtz",
                                      "bltzal"
                                      "bgezal"
                                      ])
        self.jumps = set(["b",
                          "bal",
                          "j",
                          "jal",
                          "jr",
                          "jalr",
                          "jalr.hb",
                          "jr.hb",
                          "jalx"
                          ])

        self.returns = set(["eret",
                            "deret",
                            ])

        self.calls = set(["syscall", ])
