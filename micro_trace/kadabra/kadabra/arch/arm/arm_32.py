from collections import OrderedDict
from unicorn import UC_ARCH_ARM, UC_MODE_ARM

from unicorn.arm_const import *


class ARM_32:
    def __init__(self):
        self.SB = "FP"
        self.SP = "SP"
        self.IP = "PC"
        self.FLAGS = "CPSR"
        self.segment_registers = set(["SB", "SL", "IP", "LR", "PC"])

        self.uc_arch = UC_ARCH_ARM
        self.uc_mode = UC_MODE_ARM

        self.size = 32

        self.registers = OrderedDict([("R0", (UC_ARM_REG_R0, 32)),
                                      ("R1", (UC_ARM_REG_R1, 32)),
                                      ("R2", (UC_ARM_REG_R2, 32)),
                                      ("R3", (UC_ARM_REG_R3, 32)),
                                      ("R4", (UC_ARM_REG_R4, 32)),
                                      ("R5", (UC_ARM_REG_R5, 32)),
                                      ("R6", (UC_ARM_REG_R6, 32)),
                                      ("R7", (UC_ARM_REG_R7, 32)),
                                      ("R8", (UC_ARM_REG_R8, 32)),

                                      ("SB", (UC_ARM_REG_SB, 32)),
                                      ("SL", (UC_ARM_REG_SL, 32)),
                                      ("FP", (UC_ARM_REG_FP, 32)),
                                      ("IP", (UC_ARM_REG_IP, 32)),
                                      ("SP", (UC_ARM_REG_SP, 32)),
                                      ("LR", (UC_ARM_REG_LR, 32)),
                                      ("PC", (UC_ARM_REG_PC, 32)),
                                      ("CPSR", (UC_ARM_REG_CPSR, 32)),
                                      ("APSR", (UC_ARM_REG_APSR, 32)),

                                      ])

        self.conditional_jumps = set(["beq",
                                      "bne",
                                      "cbz",
                                      "cbnz",
                                      "bcc",
                                      "bcs",
                                      "bge",
                                      "bgt",
                                      "bl-lo",
                                      "ble",
                                      "bls",
                                      "blt",
                                      "bmi",
                                      "bpl",
                                      "bvc",
                                      "bx-hs",
                                      "bx-rs",
                                      "swi",
                                      "bhi",
                                      "bvs",

                                      ])
        self.jumps = set(["b",
                          "bx",
                          "bxj",
                          "jz",
                          "bl",
                          "blx"
                          ])

        self.returns = set(["ret",
                            "reti",
                            "c2",
                            "ca",])

        self.calls = set(["bl",
                          "blx"])
