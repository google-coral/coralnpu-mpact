.text
.globl _start
_start:
    fcvt.bf16.s f2, f1, dyn
    fcvt.bf16.s f2, f1, rne
    # Invalid static rounding mode (rm=5) for illegal instruction testing.
    .insn 4, 0x44805153
    fcvt.bf16.s f2, f1, rup
    ebreak

