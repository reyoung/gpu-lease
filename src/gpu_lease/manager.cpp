#include "gpu_lease/manager.h"

#include "gpu_lease/common.h"

#include <algorithm>

namespace gpu_lease {

LeaseManager::LeaseManager() = default;

LeaseManager::LeaseManager(std::vector<int> available_ids)
    : available_ids_(NormalizeIDs(available_ids)) {}

LeaseSelection LeaseManager::SelectIDs(const std::vector<int>& ids) const {
  LeaseSelection result;
  if (ids.empty()) {
    result.error = "ids must not be empty";
    return result;
  }

  const auto [held_id, holder] = FirstHeld(ids);
  if (!holder.empty()) {
    result.waitable = true;
    result.error =
        "GPU " + std::to_string(held_id) + " is already leased by " + holder;
    return result;
  }

  result.ok = true;
  result.ids = ids;
  return result;
}

LeaseSelection LeaseManager::SelectAny(int count) const {
  LeaseSelection result;
  if (count <= 0) {
    result.error = "count must be greater than 0";
    return result;
  }
  if (available_ids_.empty()) {
    result.error = "no GPU inventory is available";
    return result;
  }

  for (const int id : available_ids_) {
    if (held_by_.find(id) != held_by_.end()) continue;
    result.ids.push_back(id);
    if (static_cast<int>(result.ids.size()) == count) break;
  }
  if (static_cast<int>(result.ids.size()) == count) {
    result.ok = true;
    return result;
  }

  result.waitable = true;
  result.error = "requested " + std::to_string(count) + " GPU(s), only " +
                 std::to_string(result.ids.size()) + " available";
  result.ids.clear();
  return result;
}

std::string LeaseManager::Assign(const std::vector<int>& ids) {
  ++next_id_;
  const std::string lease_id = "lease-" + std::to_string(next_id_);
  for (const int id : ids) {
    held_by_[id] = lease_id;
  }
  return lease_id;
}

std::vector<int> LeaseManager::Release(const std::string& lease_id) {
  std::vector<int> released;
  for (auto it = held_by_.begin(); it != held_by_.end();) {
    if (it->second == lease_id) {
      released.push_back(it->first);
      it = held_by_.erase(it);
    } else {
      ++it;
    }
  }
  return released;
}

std::map<std::string, std::string> LeaseManager::Status() const {
  std::map<std::string, std::string> out;
  for (const auto& [id, holder] : held_by_) {
    out[std::to_string(id)] = holder;
  }
  return out;
}

bool LeaseManager::IsHeld(int id) const {
  return held_by_.find(id) != held_by_.end();
}

std::pair<int, std::string> LeaseManager::FirstHeld(
    const std::vector<int>& ids) const {
  for (const int id : ids) {
    const auto it = held_by_.find(id);
    if (it != held_by_.end()) {
      return {id, it->second};
    }
  }
  return {0, ""};
}

}  // namespace gpu_lease
