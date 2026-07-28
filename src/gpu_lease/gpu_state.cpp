#include "gpu_lease/gpu_state.h"

#include <nvml.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <dlfcn.h>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace gpu_lease {
namespace {

class Nvml final {
 public:
  explicit Nvml(std::string* error) {
    handle_ = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) {
      handle_ = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (handle_ == nullptr) {
      *error = dlerror();
      return;
    }

    init_ = Symbol<nvmlReturn_t (*)()>("nvmlInit_v2", error);
    shutdown_ = Symbol<nvmlReturn_t (*)()>("nvmlShutdown", error);
    error_string_ =
        Symbol<const char* (*)(nvmlReturn_t)>("nvmlErrorString", error);
    get_handle_by_index_ =
        Symbol<nvmlReturn_t (*)(unsigned int, nvmlDevice_t*)>(
            "nvmlDeviceGetHandleByIndex_v2", error);
    get_memory_info_ =
        Symbol<nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_t*)>(
            "nvmlDeviceGetMemoryInfo", error);
    get_utilization_rates_ =
        Symbol<nvmlReturn_t (*)(nvmlDevice_t, nvmlUtilization_t*)>(
            "nvmlDeviceGetUtilizationRates", error);
    get_compute_processes_ =
        Symbol<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*, nvmlProcessInfo_t*)>(
            "nvmlDeviceGetComputeRunningProcesses_v3", error);
    if (get_compute_processes_ == nullptr) {
      error->clear();
      get_compute_processes_ =
          Symbol<nvmlReturn_t (*)(nvmlDevice_t, unsigned int*,
                                  nvmlProcessInfo_t*)>(
              "nvmlDeviceGetComputeRunningProcesses_v2", error);
    }
    if (!error->empty()) return;

    const nvmlReturn_t status = init_();
    if (status != NVML_SUCCESS) {
      *error = "nvmlInit_v2: " + ErrorString(status);
      return;
    }
    initialized_ = true;
  }

  ~Nvml() {
    if (initialized_ && shutdown_ != nullptr) {
      shutdown_();
    }
    if (handle_ != nullptr) {
      dlclose(handle_);
    }
  }

  Nvml(const Nvml&) = delete;
  Nvml& operator=(const Nvml&) = delete;

  bool ok() const { return initialized_; }

  std::string ErrorString(nvmlReturn_t status) const {
    if (error_string_ == nullptr) {
      return std::to_string(static_cast<int>(status));
    }
    return error_string_(status);
  }

  bool Device(int id, nvmlDevice_t* device, std::string* error) const {
    if (id < 0) {
      *error = "invalid negative GPU id " + std::to_string(id);
      return false;
    }
    const nvmlReturn_t status =
        get_handle_by_index_(static_cast<unsigned int>(id), device);
    if (status != NVML_SUCCESS) {
      *error = "GPU " + std::to_string(id) +
               ": nvmlDeviceGetHandleByIndex_v2: " + ErrorString(status);
      return false;
    }
    return true;
  }

  bool Memory(nvmlDevice_t device, nvmlMemory_t* memory,
              std::string* error) const {
    memset(memory, 0, sizeof(*memory));
    const nvmlReturn_t status = get_memory_info_(device, memory);
    if (status != NVML_SUCCESS) {
      *error = "nvmlDeviceGetMemoryInfo: " + ErrorString(status);
      return false;
    }
    return true;
  }

  bool Utilization(nvmlDevice_t device, nvmlUtilization_t* utilization,
                   std::string* error) const {
    memset(utilization, 0, sizeof(*utilization));
    const nvmlReturn_t status = get_utilization_rates_(device, utilization);
    if (status != NVML_SUCCESS) {
      *error = "nvmlDeviceGetUtilizationRates: " + ErrorString(status);
      return false;
    }
    return true;
  }

  bool ComputeProcesses(nvmlDevice_t device, std::vector<unsigned int>* pids,
                        std::string* error) const {
    pids->clear();
    unsigned int count = 0;
    nvmlReturn_t status =
        get_compute_processes_(device, &count, nullptr);
    if (status == NVML_SUCCESS && count == 0) {
      return true;
    }
    if (status != NVML_ERROR_INSUFFICIENT_SIZE && status != NVML_SUCCESS) {
      *error = "nvmlDeviceGetComputeRunningProcesses: " + ErrorString(status);
      return false;
    }

    for (int attempt = 0; attempt < 3; ++attempt) {
      std::vector<nvmlProcessInfo_t> infos(count + 8);
      unsigned int actual = static_cast<unsigned int>(infos.size());
      status = get_compute_processes_(device, &actual, infos.data());
      if (status == NVML_ERROR_INSUFFICIENT_SIZE) {
        count = actual + 8;
        continue;
      }
      if (status != NVML_SUCCESS) {
        *error =
            "nvmlDeviceGetComputeRunningProcesses: " + ErrorString(status);
        return false;
      }
      for (unsigned int i = 0; i < actual; ++i) {
        pids->push_back(infos[i].pid);
      }
      return true;
    }

    *error = "nvmlDeviceGetComputeRunningProcesses: process list changed too "
             "quickly";
    return false;
  }

 private:
  template <typename T>
  T Symbol(const char* name, std::string* error) {
    dlerror();
    void* symbol = dlsym(handle_, name);
    const char* dl_error = dlerror();
    if (dl_error != nullptr) {
      *error = std::string(name) + ": " + dl_error;
      return nullptr;
    }
    return reinterpret_cast<T>(symbol);
  }

  void* handle_ = nullptr;
  bool initialized_ = false;
  nvmlReturn_t (*init_)() = nullptr;
  nvmlReturn_t (*shutdown_)() = nullptr;
  const char* (*error_string_)(nvmlReturn_t) = nullptr;
  nvmlReturn_t (*get_handle_by_index_)(unsigned int, nvmlDevice_t*) = nullptr;
  nvmlReturn_t (*get_memory_info_)(nvmlDevice_t, nvmlMemory_t*) = nullptr;
  nvmlReturn_t (*get_utilization_rates_)(nvmlDevice_t,
                                         nvmlUtilization_t*) = nullptr;
  nvmlReturn_t (*get_compute_processes_)(nvmlDevice_t, unsigned int*,
                                         nvmlProcessInfo_t*) = nullptr;
};

Nvml* SharedNvml(std::string* error) {
  // The NVIDIA driver keeps internal eventfds alive across
  // nvmlShutdown()/dlclose().  Constructing a fresh Nvml object for every
  // lease therefore leaked one daemon FD per acquire/release cycle and
  // eventually caused EMFILE.  The daemon is single-process and serializes
  // lease preparation, so one process-lifetime NVML session is sufficient.
  static std::string initialization_error;
  static Nvml nvml(&initialization_error);
  if (!nvml.ok()) {
    *error = initialization_error;
    return nullptr;
  }
  return &nvml;
}

std::string ProcExeBasename(unsigned int pid) {
  const std::string path = "/proc/" + std::to_string(pid) + "/exe";
  std::vector<char> buffer(4096);
  const ssize_t n = readlink(path.c_str(), buffer.data(), buffer.size() - 1);
  if (n <= 0) return "";
  buffer[static_cast<size_t>(n)] = '\0';
  std::string exe(buffer.data());
  const size_t slash = exe.find_last_of('/');
  return slash == std::string::npos ? exe : exe.substr(slash + 1);
}

bool IsGpuLeaseProcess(unsigned int pid) {
  if (pid == static_cast<unsigned int>(getpid())) return true;
  return ProcExeBasename(pid) == "gpu-lease";
}

bool IsAlive(unsigned int pid) {
  return kill(static_cast<pid_t>(pid), 0) == 0 || errno == EPERM;
}

bool KillPid(unsigned int pid, int signal_number, std::ostream& err) {
  if (kill(static_cast<pid_t>(pid), signal_number) == 0) {
    return true;
  }
  if (errno == ESRCH) {
    return true;
  }
  err << "lease: failed to kill GPU process " << pid << ": "
      << strerror(errno) << "\n";
  return false;
}

bool KillUnmanagedProcessesOnDevice(const Nvml& nvml, int id,
                                    nvmlDevice_t device, std::ostream& err) {
  std::vector<unsigned int> pids;
  std::string error;
  if (!nvml.ComputeProcesses(device, &pids, &error)) {
    err << "lease: GPU " << id << ": " << error << "\n";
    return false;
  }

  std::vector<unsigned int> targets;
  for (const unsigned int pid : pids) {
    if (!IsGpuLeaseProcess(pid)) targets.push_back(pid);
  }
  if (targets.empty()) return true;

  for (const unsigned int pid : targets) {
    err << "lease: killing unmanaged GPU process " << pid << " on GPU " << id
        << "\n";
    if (!KillPid(pid, SIGTERM, err)) return false;
  }

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline) {
    bool any_alive = false;
    for (const unsigned int pid : targets) {
      if (IsAlive(pid)) {
        any_alive = true;
        break;
      }
    }
    if (!any_alive) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  for (const unsigned int pid : targets) {
    if (IsAlive(pid) && !KillPid(pid, SIGKILL, err)) return false;
  }
  return true;
}

bool DeviceIsIdle(const Nvml& nvml, int id, nvmlDevice_t device,
                  std::ostream& err, std::string* busy_reason) {
  nvmlMemory_t memory;
  std::string error;
  if (!nvml.Memory(device, &memory, &error)) {
    err << "lease: GPU " << id << ": " << error << "\n";
    return false;
  }

  nvmlUtilization_t utilization;
  if (!nvml.Utilization(device, &utilization, &error)) {
    err << "lease: GPU " << id << ": " << error << "\n";
    return false;
  }

  if (utilization.gpu != 0) {
    std::ostringstream reason;
    reason << "GPU " << id << " memory.used=" << memory.used
           << " gpu_util=" << utilization.gpu << "%";
    *busy_reason = reason.str();
    return true;
  }

  std::vector<unsigned int> pids;
  if (!nvml.ComputeProcesses(device, &pids, &error)) {
    err << "lease: GPU " << id << ": " << error << "\n";
    return false;
  }
  for (const unsigned int pid : pids) {
    if (!IsGpuLeaseProcess(pid)) {
      *busy_reason = "GPU " + std::to_string(id) +
                     " still has unmanaged compute process " +
                     std::to_string(pid);
      return true;
    }
  }

  if (memory.used != 0 && pids.empty()) {
    busy_reason->clear();
    return true;
  }

  busy_reason->clear();
  return true;
}

bool EnvDisablesPreparation() {
  const char* value = getenv("GPU_LEASE_DISABLE_PRESTART_CHECK");
  return value != nullptr && value[0] != '\0' && std::string(value) != "0";
}

int EnvInt(const char* name, int fallback) {
  const char* value = getenv(name);
  if (value == nullptr || value[0] == '\0') return fallback;
  char* end = nullptr;
  const long parsed = strtol(value, &end, 10);
  if (end == value || *end != '\0' || parsed <= 0 || parsed > 3600000) {
    return fallback;
  }
  return static_cast<int>(parsed);
}

}  // namespace

bool PrepareLeasedGPUs(const std::vector<int>& ids,
                       const PrepareGPUsOptions& options, std::ostream& err) {
  if (ids.empty() || EnvDisablesPreparation()) return true;

  PrepareGPUsOptions effective = options;
  effective.timeout_ms =
      EnvInt("GPU_LEASE_PRESTART_TIMEOUT_MS", effective.timeout_ms);
  effective.poll_ms = EnvInt("GPU_LEASE_PRESTART_POLL_MS", effective.poll_ms);

  std::string error;
  Nvml* nvml = SharedNvml(&error);
  if (nvml == nullptr) {
    err << "lease: initialize NVML: " << error << "\n";
    return false;
  }

  std::vector<nvmlDevice_t> devices;
  devices.reserve(ids.size());
  for (const int id : ids) {
    nvmlDevice_t device = nullptr;
    if (!nvml->Device(id, &device, &error)) {
      err << "lease: " << error << "\n";
      return false;
    }
    devices.push_back(device);
  }

  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(effective.timeout_ms);
  std::string last_busy_reason;
  while (true) {
    if (effective.kill_unmanaged_processes) {
      for (size_t i = 0; i < devices.size(); ++i) {
        if (!KillUnmanagedProcessesOnDevice(*nvml, ids[i], devices[i], err)) {
          return false;
        }
      }
    }

    bool all_idle = true;
    if (effective.wait_until_idle) {
      for (size_t i = 0; i < devices.size(); ++i) {
        std::string busy_reason;
        if (!DeviceIsIdle(*nvml, ids[i], devices[i], err, &busy_reason)) {
          return false;
        }
        if (!busy_reason.empty()) {
          all_idle = false;
          last_busy_reason = busy_reason;
        }
      }
    }

    if (all_idle) return true;
    if (std::chrono::steady_clock::now() >= deadline) {
      err << "lease: timed out waiting for leased GPUs to become idle";
      if (!last_busy_reason.empty()) err << ": " << last_busy_reason;
      err << "\n";
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(effective.poll_ms));
  }
}

}  // namespace gpu_lease
