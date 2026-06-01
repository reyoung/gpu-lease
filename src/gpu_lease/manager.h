#pragma once

#include <map>
#include <string>
#include <vector>

namespace gpu_lease {

struct LeaseSelection {
  bool ok = false;
  bool waitable = false;
  std::string error;
  std::vector<int> ids;
};

class LeaseManager {
 public:
  LeaseManager();
  explicit LeaseManager(std::vector<int> available_ids);

  LeaseSelection SelectIDs(const std::vector<int>& ids) const;
  LeaseSelection SelectAny(int count) const;
  std::string Assign(const std::vector<int>& ids);
  std::vector<int> Release(const std::string& lease_id);

  std::map<std::string, std::string> Status() const;
  bool IsHeld(int id) const;
  const std::vector<int>& AvailableIDs() const { return available_ids_; }

 private:
  std::pair<int, std::string> FirstHeld(const std::vector<int>& ids) const;

  std::map<int, std::string> held_by_;
  std::vector<int> available_ids_;
  unsigned long long next_id_ = 0;
};

}  // namespace gpu_lease
