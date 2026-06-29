#pragma once

#include <iosfwd>
#include <vector>

namespace gpu_lease {

struct PrepareGPUsOptions {
  bool kill_unmanaged_processes = true;
  bool wait_until_idle = true;
  int timeout_ms = 30000;
  int poll_ms = 100;
};

bool PrepareLeasedGPUs(const std::vector<int>& ids,
                       const PrepareGPUsOptions& options, std::ostream& err);

}  // namespace gpu_lease
