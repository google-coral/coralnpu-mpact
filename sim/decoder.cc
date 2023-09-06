#include "sim/decoder.h"

#include "mpact/sim/generic/type_helpers.h"

namespace kelvin::sim {

using ::mpact::sim::generic::operator*;  // NOLINT: is used below (clang error).

KelvinDecoder::KelvinDecoder(KelvinState *state,
                             mpact::sim::util::MemoryInterface *memory)
    : state_(state), memory_(memory) {
  // Get a handle to the internal error in the program error controller.
  decode_error_ = state->program_error_controller()->GetProgramError(
      mpact::sim::generic::ProgramErrorController::kInternalErrorName);

  // Need a data buffer to load instructions from memory. Allocate a single
  // buffer that can be reused for each instruction word.
  inst_db_ = state_->db_factory()->Allocate<uint32_t>(1);
  // Allocate the isa factory class, the top level isa decoder
  // instance, and the encoding parser.
  kelvin_isa_factory_ = new KelvinIsaFactory();
  kelvin_isa_ = new isa32::KelvinInstructionSet(state, kelvin_isa_factory_);
  kelvin_encoding_ = new isa32::KelvinEncoding(state);
  decode_error_ = state->program_error_controller()->GetProgramError(
      mpact::sim::generic::ProgramErrorController::kInternalErrorName);
}

KelvinDecoder::~KelvinDecoder() {
  inst_db_->DecRef();
  delete kelvin_isa_;
  delete kelvin_isa_factory_;
  delete kelvin_encoding_;
}

mpact::sim::generic::Instruction *KelvinDecoder::DecodeInstruction(
    uint64_t address) {
  // First check that the address is aligned properly. If not, create and return
  // an empty instruction object and raise an exception.
  if (address & 0x1) {
    auto *inst = new mpact::sim::generic::Instruction(address, state_);
    inst->set_semantic_function(
        [](mpact::sim::generic::Instruction *inst) { /* empty */ });
    inst->set_size(1);
    inst->SetDisassemblyString("Misaligned instruction address");
    inst->set_opcode(static_cast<int>(isa32::OpcodeEnum::kNone));
    state_->Trap(
        /*is_interrupt*/ false, address,
        *mpact::sim::riscv::ExceptionCode::kInstructionAddressMisaligned,
        address ^ 0x1, inst);
    return inst;
  }

  // If the address is greater than the max address, raise an exception.
  if (address > state_->max_physical_address()) {
    state_->Trap(/*is_interrupt*/ false, address,
                 *mpact::sim::riscv::ExceptionCode::kInstructionAccessFault,
                 address, nullptr);
    auto *inst = new mpact::sim::generic::Instruction(address, state_);
    inst->set_size(0);
    inst->SetDisassemblyString("Instruction access fault");
    inst->set_opcode(static_cast<int>(isa32::OpcodeEnum::kNone));
    inst->set_semantic_function(
        [](mpact::sim::generic::Instruction *inst) { /* empty */ });
    return inst;
  }

  // Read the instruction word from memory and parse it in the encoding parser.
  memory_->Load(address, inst_db_, nullptr, nullptr);
  auto iword = inst_db_->Get<uint32_t>(0);
  kelvin_encoding_->ParseInstruction(iword);

  // Call the isa decoder to obtain a new instruction object for the instruction
  // word that was parsed above.
  auto *instruction = kelvin_isa_->Decode(address, kelvin_encoding_);
  return instruction;
}
}  // namespace kelvin::sim
