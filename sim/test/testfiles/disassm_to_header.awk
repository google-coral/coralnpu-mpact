#!/usr/bin/awk -f

# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# AWK script to transform disassembly into a C++ vector of instructions.
#
# Usage:
#   awk -f disassm_to_header.awk your_disassembly_file.txt > output.h

BEGIN {
    if (!header_guard) {
        print "Error: header_guard variable not set." > "/dev/stderr"
        exit 1
    }
    if (!nested_namespace) {
        print "Error: nested_namespace variable not set." > "/dev/stderr"
        exit 1
    }
    print "#ifndef " header_guard
    print "#define " header_guard
    print ""
    print "#include <cstdint>"
    print "#include <vector>"
    print ""
    print "namespace coralnpu::sim::test_data::" nested_namespace " {"
    print ""
    print "struct InstructionData {"
    print "  uint32_t address;"
    print "  uint32_t instruction;"
    print "};"
    print ""
    print "inline const std::vector<InstructionData> &GetInstructions() {"
    print "  static const auto* instructions = new std::vector<InstructionData>({"
}

(NF > 2 && $1 ~ /^[0-9a-f]+:$/) {
    pc = $1;
    sub(/:$/, "", pc);
    printf "    {0x%s, 0x%s},\n", pc, $2;
}

END {
    print "  });"
    print "  return *instructions;"
    print "}"
    print ""
    print "}  // namespace coralnpu::sim::test_data::" nested_namespace
    print ""
    print "#endif  // " header_guard
}
