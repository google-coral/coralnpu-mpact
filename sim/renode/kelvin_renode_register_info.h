#ifndef LEARNING_BRAIN_RESEARCH_KELVIN_SIM_RENODE_KELVIN_RENODE_REGISTER_INFO_H_
#define LEARNING_BRAIN_RESEARCH_KELVIN_SIM_RENODE_KELVIN_RENODE_REGISTER_INFO_H_

#include <vector>

#include "sim/renode/renode_debug_interface.h"

namespace kelvin::sim {

// This file defines a class that is used to store the register information
// that needs to be provided to renode for the Kelvin registers.

class KelvinRenodeRegisterInfo {
 public:
  using RenodeRegisterInfo = std::vector<renode::RenodeCpuRegister>;
  static const RenodeRegisterInfo &GetRenodeRegisterInfo();

 private:
  KelvinRenodeRegisterInfo();
  static KelvinRenodeRegisterInfo *Instance();
  void InitializeRenodeRegisterInfo();
  const RenodeRegisterInfo &GetRenodeRegisterInfoPrivate();

  static KelvinRenodeRegisterInfo *instance_;
  RenodeRegisterInfo renode_register_info_;
};

}  // namespace kelvin::sim

#endif  // LEARNING_BRAIN_RESEARCH_KELVIN_SIM_RENODE_KELVIN_RENODE_REGISTER_INFO_H_
