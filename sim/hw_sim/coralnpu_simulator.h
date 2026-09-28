#ifndef SIM_HW_SIM_CORALNPU_SIMULATOR_H_
#define SIM_HW_SIM_CORALNPU_SIMULATOR_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>

struct CoralNPUMailbox {
  uint32_t message[4] = {0, 0, 0, 0};
};

class CoralMemoryTarget {
 public:
  virtual ~CoralMemoryTarget() = default;

  virtual void Load(uint64_t address, uint8_t* data, size_t size) = 0;
  virtual void Store(uint64_t address, const uint8_t* data, size_t size) = 0;
};

// callback for tracing
using TraceCallback = std::function<void(uint32_t pc, uint32_t instruction,
                                         std::string& disassembly)>;

class CoralNPUSimulator {
 public:
  virtual ~CoralNPUSimulator() = default;

  // Functions for reading/writing memory and Mailbox.
  virtual void ReadMem(uint32_t addr, size_t size, char* data) = 0;
  virtual const CoralNPUMailbox& ReadMailbox() = 0;
  virtual void WriteMem(uint32_t addr, size_t size, const char* data) = 0;
  virtual void WriteMailbox(const CoralNPUMailbox& mailbox) = 0;

  // Wait for interrupt
  virtual bool WaitForTermination(int timeout) = 0;

  // Begin executing starting with the PC set to the specified address. Returns
  // when the core halts.
  virtual void Run(uint32_t start_addr) = 0;

  // Returns the total simulated cycle count.
  virtual uint64_t GetCycleCount() const = 0;

  // Set a callback for tracing instructions with optional disassembly.
  virtual void SetTraceCallback(TraceCallback callback, bool disasm) = 0;
};

extern "C" CoralNPUSimulator* coralnpu_simulator_mpact_create(void);

#endif  // SIM_HW_SIM_CORALNPU_SIMULATOR_H_
