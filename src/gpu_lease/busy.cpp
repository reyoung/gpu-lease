#include "gpu_lease/busy.h"

#include "gpu_lease/manager.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <sstream>
#include <thread>
#include <utility>

namespace gpu_lease {
namespace {

std::string CudaErrorString(cudaError_t err) {
  return std::string(cudaGetErrorString(err));
}

std::string CublasErrorString(cublasStatus_t status) {
  return "cuBLAS status " + std::to_string(static_cast<int>(status));
}

volatile sig_atomic_t g_child_stop = 0;

bool WriteAllFd(int fd, const std::string& data);

class CublasBusyWorker final : public BusyWorker {
 public:
  CublasBusyWorker(pid_t pid, int stop_fd) : pid_(pid), stop_fd_(stop_fd) {}

  ~CublasBusyWorker() override { Stop(); }

  void Stop() override {
    if (pid_ <= 0) return;

    if (stop_fd_ >= 0) {
      (void)WriteAllFd(stop_fd_, "x");
      close(stop_fd_);
      stop_fd_ = -1;
    }

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (std::chrono::steady_clock::now() < deadline) {
      int status = 0;
      const pid_t got = waitpid(pid_, &status, WNOHANG);
      if (got == pid_) {
        pid_ = -1;
        return;
      }
      if (got < 0 && errno == ECHILD) {
        pid_ = -1;
        return;
      }
      if (got < 0 && errno != EINTR) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    (void)kill(pid_, SIGTERM);
    const auto term_deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (std::chrono::steady_clock::now() < term_deadline) {
      int status = 0;
      const pid_t got = waitpid(pid_, &status, WNOHANG);
      if (got == pid_ || (got < 0 && errno == ECHILD)) {
        pid_ = -1;
        return;
      }
      if (got < 0 && errno != EINTR) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    (void)kill(pid_, SIGKILL);
    while (true) {
      int status = 0;
      const pid_t got = waitpid(pid_, &status, 0);
      if (got == pid_ || (got < 0 && errno == ECHILD)) break;
      if (got < 0 && errno == EINTR) continue;
      break;
    }
    pid_ = -1;
  }

 private:
  pid_t pid_ = -1;
  int stop_fd_ = -1;
};

bool WriteAllFd(int fd, const std::string& data) {
  size_t offset = 0;
  while (offset < data.size()) {
    const ssize_t n = write(fd, data.data() + offset, data.size() - offset);
    if (n > 0) {
      offset += static_cast<size_t>(n);
      continue;
    }
    if (n < 0 && errno == EINTR) continue;
    return false;
  }
  return true;
}

std::string ReadAllFd(int fd) {
  std::string out;
  char buffer[512];
  while (true) {
    const ssize_t n = read(fd, buffer, sizeof(buffer));
    if (n > 0) {
      out.append(buffer, static_cast<size_t>(n));
      continue;
    }
    if (n == 0) break;
    if (errno == EINTR) continue;
    break;
  }
  return out;
}

void CloseInheritedFdsExcept(int keep_fd0, int keep_fd1) {
  long max_fd = sysconf(_SC_OPEN_MAX);
  if (max_fd < 0) max_fd = 4096;
  for (int fd = 3; fd < max_fd; ++fd) {
    if (fd == keep_fd0 || fd == keep_fd1) continue;
    close(fd);
  }
}

void BusyChildSignalHandler(int) { g_child_stop = 1; }

bool StopRequested(int stop_fd) {
  if (g_child_stop) return true;
  char buffer[16];
  while (true) {
    const ssize_t n = read(stop_fd, buffer, sizeof(buffer));
    if (n > 0 || n == 0) return true;
    if (errno == EINTR) continue;
    if (errno == EAGAIN || errno == EWOULDBLOCK) return false;
    return true;
  }
}

int RunCublasWorkerChild(int gpu_id, int init_fd, int stop_fd) {
  CloseInheritedFdsExcept(init_fd, stop_fd);
  signal(SIGTERM, BusyChildSignalHandler);
  signal(SIGINT, BusyChildSignalHandler);

  cudaStream_t stream = nullptr;
  cublasHandle_t handle = nullptr;
  float* a = nullptr;
  float* b = nullptr;
  float* c = nullptr;
  bool device_set = false;

  auto cleanup = [&]() {
    if (a != nullptr) cudaFree(a);
    if (b != nullptr) cudaFree(b);
    if (c != nullptr) cudaFree(c);
    if (handle != nullptr) cublasDestroy(handle);
    if (stream != nullptr) cudaStreamDestroy(stream);
    if (device_set) cudaDeviceReset();
  };

  auto fail = [&](const std::string& msg) {
    cleanup();
    (void)WriteAllFd(init_fd, "ERR " + msg);
    close(init_fd);
    return 1;
  };

  cudaError_t cuda_status = cudaSetDevice(gpu_id);
  if (cuda_status != cudaSuccess) {
    return fail("cudaSetDevice(" + std::to_string(gpu_id) + "): " +
                CudaErrorString(cuda_status));
  }
  device_set = true;

  cuda_status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (cuda_status != cudaSuccess) {
    return fail("cudaStreamCreateWithFlags: " + CudaErrorString(cuda_status));
  }

  cublasStatus_t cublas_status = cublasCreate(&handle);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    return fail("cublasCreate: " + CublasErrorString(cublas_status));
  }
  cublas_status = cublasSetStream(handle, stream);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    return fail("cublasSetStream: " + CublasErrorString(cublas_status));
  }

  constexpr int n = 1024;
  constexpr size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
  cuda_status = cudaMalloc(&a, bytes);
  if (cuda_status != cudaSuccess) {
    return fail("cudaMalloc A: " + CudaErrorString(cuda_status));
  }
  cuda_status = cudaMalloc(&b, bytes);
  if (cuda_status != cudaSuccess) {
    return fail("cudaMalloc B: " + CudaErrorString(cuda_status));
  }
  cuda_status = cudaMalloc(&c, bytes);
  if (cuda_status != cudaSuccess) {
    return fail("cudaMalloc C: " + CudaErrorString(cuda_status));
  }

  cudaMemsetAsync(a, 1, bytes, stream);
  cudaMemsetAsync(b, 2, bytes, stream);
  cudaMemsetAsync(c, 0, bytes, stream);
  cuda_status = cudaStreamSynchronize(stream);
  if (cuda_status != cudaSuccess) {
    return fail("initial cudaStreamSynchronize: " + CudaErrorString(cuda_status));
  }

  (void)WriteAllFd(init_fd, "OK");
  close(init_fd);

  const float alpha = 1.0f;
  const float beta = 0.0f;
  while (!StopRequested(stop_fd)) {
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
      break;
    }
    cublas_status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                &alpha, a, n, b, n, &beta, c, n);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      break;
    }
    cuda_status = cudaStreamSynchronize(stream);
    if (cuda_status != cudaSuccess) {
      break;
    }
  }

  cleanup();
  close(stop_fd);
  return 0;
}

std::unique_ptr<BusyWorker> StartCublasWorker(int gpu_id, std::string* error) {
  int init_pipe[2];
  if (pipe2(init_pipe, O_CLOEXEC) != 0) {
    *error = "pipe2: " + std::string(strerror(errno));
    return nullptr;
  }
  int stop_pipe[2];
  if (pipe2(stop_pipe, O_CLOEXEC | O_NONBLOCK) != 0) {
    *error = "pipe2: " + std::string(strerror(errno));
    close(init_pipe[0]);
    close(init_pipe[1]);
    return nullptr;
  }

  const pid_t pid = fork();
  if (pid < 0) {
    *error = "fork: " + std::string(strerror(errno));
    close(init_pipe[0]);
    close(init_pipe[1]);
    close(stop_pipe[0]);
    close(stop_pipe[1]);
    return nullptr;
  }

  if (pid == 0) {
    close(init_pipe[0]);
    close(stop_pipe[1]);
    const int code = RunCublasWorkerChild(gpu_id, init_pipe[1], stop_pipe[0]);
    _exit(code);
  }

  close(init_pipe[1]);
  close(stop_pipe[0]);
  const std::string init = ReadAllFd(init_pipe[0]);
  close(init_pipe[0]);
  if (init != "OK") {
    close(stop_pipe[1]);
    int status = 0;
    (void)waitpid(pid, &status, 0);
    if (init.rfind("ERR ", 0) == 0) {
      *error = init.substr(4);
    } else {
      *error = "busy worker exited before initialization completed";
    }
    return nullptr;
  }

  return std::make_unique<CublasBusyWorker>(pid, stop_pipe[1]);
}

}  // namespace

std::unique_ptr<BusyWorker> CudaBusyBackend::Start(int gpu_id,
                                                  std::string* error) {
  return StartCublasWorker(gpu_id, error);
}

BusyController::BusyController(std::vector<int> managed_ids,
                               BusyBackend* backend, Logger logger)
    : managed_ids_(managed_ids.begin(), managed_ids.end()),
      backend_(backend),
      logger_(std::move(logger)) {}

BusyController::~BusyController() { StopAll(); }

void BusyController::StartIdle(const LeaseManager& manager) {
  for (const int id : managed_ids_) {
    if (!manager.IsHeld(id)) {
      StartOne(id);
    }
  }
}

void BusyController::StopForLease(const std::vector<int>& ids) {
  for (const int id : ids) {
    const auto it = workers_.find(id);
    if (it == workers_.end()) continue;
    it->second->Stop();
    workers_.erase(it);
  }
}

void BusyController::RestartReleased(const std::vector<int>& ids,
                                     const LeaseManager& manager) {
  for (const int id : ids) {
    if (Manages(id) && !manager.IsHeld(id)) {
      StartOne(id);
    }
  }
}

void BusyController::StopAll() {
  for (auto& [_, worker] : workers_) {
    worker->Stop();
  }
  workers_.clear();
}

bool BusyController::Manages(int gpu_id) const {
  return managed_ids_.find(gpu_id) != managed_ids_.end();
}

void BusyController::StartOne(int gpu_id) {
  if (!Manages(gpu_id) || workers_.find(gpu_id) != workers_.end()) {
    return;
  }
  std::string error;
  std::unique_ptr<BusyWorker> worker = backend_->Start(gpu_id, &error);
  if (!worker) {
    if (logger_) {
      logger_("busy-matmul: GPU " + std::to_string(gpu_id) + ": " + error);
    }
    return;
  }
  workers_[gpu_id] = std::move(worker);
}

}  // namespace gpu_lease
