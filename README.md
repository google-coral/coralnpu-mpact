# CoralNPU Instruction Simulator

This project contains the instruction simulator of CoralNPU ML core based on
[MPACT-Sim](https://github.com/google/mpact-sim) and
[MPACT-RiscV](https://github.com/google/mpact-riscv). The simulator supports
RISC-V 32im configuration + CoralNPU-specific SIMD instructions. Please review
[ISA Spec](https://opensecura.googlesource.com/sw/kelvin/+/master/docs/kelvin_isa.md)
for more detail

## Project structure

```
sim         Simulator implementations
  |
  ˪ proto   Trace dump protobuf definition.
  |
  ˪ renode  Renode(https://github.com/renode/renode) integration interface
  |
  ˪ test    Simulated instruction / Framework function unit tests
```

## Build simulator

To build all targets, run

```bash
bazel build //...
```

Specifically, the simulator standalone binary can be built with

```bash
bazel build //sim:coralnpu_sim
```
