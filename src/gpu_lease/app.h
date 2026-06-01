#pragma once

#include <iosfwd>
#include <string>
#include <vector>

namespace gpu_lease {

int RunCLI(const std::vector<std::string>& args, std::ostream& out,
           std::ostream& err);

}  // namespace gpu_lease
