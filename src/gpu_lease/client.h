#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace gpu_lease {

struct AcquireOptions {
  std::vector<int> ids;
  int count = 0;
  bool wait = false;
};

class HeldLease {
 public:
  HeldLease(std::string id, std::vector<int> ids, int fd);
  ~HeldLease();

  HeldLease(const HeldLease&) = delete;
  HeldLease& operator=(const HeldLease&) = delete;
  HeldLease(HeldLease&& other) noexcept;
  HeldLease& operator=(HeldLease&& other) noexcept;

  void Close();

  const std::string& id() const { return id_; }
  const std::vector<int>& ids() const { return ids_; }

 private:
  std::string id_;
  std::vector<int> ids_;
  int fd_ = -1;
};

struct AcquireResult {
  std::unique_ptr<HeldLease> lease;
  std::string error;
};

struct StatusResult {
  std::map<std::string, std::string> leases;
  std::string error;
};

AcquireResult AcquireWithOptions(const std::string& socket_path,
                                 const AcquireOptions& options);
AcquireResult Acquire(const std::string& socket_path,
                      const std::vector<int>& ids);
StatusResult GetStatus(const std::string& socket_path);

}  // namespace gpu_lease
