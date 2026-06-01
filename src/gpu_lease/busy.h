#pragma once

#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace gpu_lease {

class LeaseManager;

class BusyWorker {
 public:
  virtual ~BusyWorker() = default;
  virtual void Stop() = 0;
};

class BusyBackend {
 public:
  virtual ~BusyBackend() = default;
  virtual std::unique_ptr<BusyWorker> Start(int gpu_id,
                                            std::string* error) = 0;
};

class CudaBusyBackend : public BusyBackend {
 public:
  std::unique_ptr<BusyWorker> Start(int gpu_id, std::string* error) override;
};

class BusyController {
 public:
  using Logger = std::function<void(const std::string&)>;

  BusyController(std::vector<int> managed_ids, BusyBackend* backend,
                 Logger logger);
  ~BusyController();

  BusyController(const BusyController&) = delete;
  BusyController& operator=(const BusyController&) = delete;

  void StartIdle(const LeaseManager& manager);
  void StopForLease(const std::vector<int>& ids);
  void RestartReleased(const std::vector<int>& ids, const LeaseManager& manager);
  void StopAll();
  bool Manages(int gpu_id) const;

 private:
  void StartOne(int gpu_id);

  std::set<int> managed_ids_;
  BusyBackend* backend_ = nullptr;
  Logger logger_;
  std::map<int, std::unique_ptr<BusyWorker>> workers_;
};

}  // namespace gpu_lease
