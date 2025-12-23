#ifndef LEARNING_BRAIN_RESEARCH_KELVIN_SIM_HW_SIM_CORALNPU_SIMULATOR_H_
#define LEARNING_BRAIN_RESEARCH_KELVIN_SIM_HW_SIM_CORALNPU_SIMULATOR_H_

#include <cstddef>
#include <cstdint>

struct CoralNPUMailbox {
  uint32_t message[4] = {0, 0, 0, 0};
};

class CoralNPUSimulator {
 public:
  static CoralNPUSimulator* Create();

  virtual ~CoralNPUSimulator() = default;

  // Functions for reading/writing TCMs and Mailbox.
  virtual void ReadTCM(uint32_t addr, size_t size, char* data) = 0;
  virtual const CoralNPUMailbox& ReadMailbox() = 0;
  virtual void WriteTCM(uint32_t addr, size_t size, const char* data) = 0;
  virtual void WriteMailbox(const CoralNPUMailbox& mailbox) = 0;

  // Wait for interrupt
  virtual bool WaitForTermination(int timeout) = 0;

  // Begin executing starting with the PC set to the specified address. Returns
  // when the core halts.
  virtual void Run(uint32_t start_addr) = 0;
};

#endif  // LEARNING_BRAIN_RESEARCH_KELVIN_SIM_HW_SIM_CORALNPU_SIMULATOR_H_
