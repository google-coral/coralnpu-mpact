#include <ios>
#include <string>

#include "sim/decoder.h"
#include "sim/kelvin_state.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "elfio/elfio.hpp"
#include "elfio/elfio_section.hpp"
#include "elfio/elfio_symbols.hpp"
#include "mpact/sim/generic/instruction.h"
#include "mpact/sim/util/memory/flat_demand_memory.h"
#include "mpact/sim/util/program_loader/elf_program_loader.h"

namespace {

using ::mpact::sim::riscv::RiscVXlen;

constexpr char kFileName[] = "hello_world_rv32imf.elf";

// The depot path to the test directory.
constexpr char kDepotPath[] = "sim/test/";

using SymbolAccessor = ELFIO::symbol_section_accessor_template<ELFIO::section>;

class KelvinDecoderTest : public testing::Test {
 protected:
  KelvinDecoderTest()
      : state_("kelvin_decoder_test", RiscVXlen::RV32),
        memory_(0),
        loader_(&memory_),
        decoder_(&state_, &memory_) {
    const std::string input_file =
        absl::StrCat(kDepotPath, "testfiles/", kFileName);
    auto result = loader_.LoadProgram(input_file);
    CHECK_OK(result.status());
    elf_reader_.load(input_file);
    auto *symtab = elf_reader_.sections[".symtab"];
    CHECK_NE(symtab, nullptr);
    symbol_accessor_ = new SymbolAccessor(elf_reader_, symtab);
  }

  ~KelvinDecoderTest() override { delete symbol_accessor_; }

  ELFIO::elfio elf_reader_;
  kelvin::sim::KelvinState state_;
  mpact::sim::util::FlatDemandMemory memory_;
  mpact::sim::util::ElfProgramLoader loader_;
  kelvin::sim::KelvinDecoder decoder_;
  SymbolAccessor *symbol_accessor_;
};

// This test is really pretty simple. It decodes the instructions in "main".
// The goal of this test is not so much to ensure that the decoder is accurate,
// but that the decoder returns a non-null instruction object for each address
// in main, and that executing this instruction does not generate an error.
TEST_F(KelvinDecoderTest, HelloWorldMain) {
  ELFIO::Elf64_Addr value;
  ELFIO::Elf_Xword size;
  unsigned char bind;
  unsigned char type;
  ELFIO::Elf_Half section_index;
  unsigned char other;
  bool success = symbol_accessor_->get_symbol("main", value, size, bind, type,
                                              section_index, other);
  ASSERT_TRUE(success);
  uint64_t address = value;
  while (address < value + size) {
    LOG(INFO) << "Address: " << std::hex << address;
    EXPECT_FALSE(state_.program_error_controller()->HasError());
    auto *inst = decoder_.DecodeInstruction(address);
    ASSERT_NE(inst, nullptr);
    inst->Execute(nullptr);
    if (state_.program_error_controller()->HasError()) {
      auto errvec = state_.program_error_controller()->GetUnmaskedErrorNames();
      for (auto &err : errvec) {
        LOG(INFO) << "Error: " << err;
        auto msgvec = state_.program_error_controller()->GetErrorMessages(err);
        for (auto &msg : msgvec) {
          LOG(INFO) << "    " << msg;
        }
      }
    }
    EXPECT_FALSE(state_.program_error_controller()->HasError());
    state_.program_error_controller()->ClearAll();
    address += inst->size();
    inst->DecRef();
    state_.AdvanceDelayLines();
  }
}

// Even with a bad address, a valid instruction object should be returned.
TEST_F(KelvinDecoderTest, BadAddress) {
  auto *inst = decoder_.DecodeInstruction(0x4321);
  ASSERT_NE(inst, nullptr);
  inst->Execute(nullptr);
  inst->DecRef();
}

}  // namespace
