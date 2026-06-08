#ifndef SIM_HW_SIM_CORALNPU_SIMULATOR_H_
#define SIM_HW_SIM_CORALNPU_SIMULATOR_H_

#include <cstddef>
#include <cstdint>

#include <string>
#include <functional>


class CoralMemoryTarget {
public:
  virtual ~CoralMemoryTarget() = default;
  
  virtual void Load(uint64_t address, uint8_t* data, size_t size) = 0;
  virtual void Store(uint64_t address, const uint8_t* data, size_t size) = 0;
};

// callback for tracing
using TraceCallback = std::function<void(uint32_t pc, uint32_t instruction, std::string &disassembly)>;


class CoralNPUSimulator {
 public:
  static CoralNPUSimulator* Create();

  virtual ~CoralNPUSimulator() = default;

  // Register callbacks for memory access for the range [address, address + size - 1].
  virtual void RegisterMemoryTarget(uint64_t address, uint64_t size, CoralMemoryTarget* target) = 0;

  // Register out-of-bounds memory access callback
  virtual void RegisterOOBMemoryTarget(CoralMemoryTarget* target) = 0;

  // Wait for interrupt
  virtual bool WaitForTermination(int timeout) = 0;

  // Begin executing starting with the PC set to the specified address. Returns
  // when the core halts.
  virtual void Run(uint32_t start_addr) = 0;

  // Set a callback for tracing instructions with optional disassembly.
  virtual void SetTraceCallback(TraceCallback callback, bool disasm) = 0;
};

#endif  // SIM_HW_SIM_CORALNPU_SIMULATOR_H_
