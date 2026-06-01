#include "gpu_lease/busy.h"

#include "gpu_lease/manager.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <atomic>
#include <future>
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

struct InitResult {
  bool ok = false;
  std::string error;
};

class CublasBusyWorker final : public BusyWorker {
 public:
  explicit CublasBusyWorker(std::thread thread,
                            std::shared_ptr<std::atomic<bool>> stop)
      : thread_(std::move(thread)), stop_(std::move(stop)) {}

  ~CublasBusyWorker() override { Stop(); }

  void Stop() override {
    if (stop_) {
      stop_->store(true, std::memory_order_relaxed);
    }
    if (thread_.joinable()) {
      thread_.join();
    }
  }

 private:
  std::thread thread_;
  std::shared_ptr<std::atomic<bool>> stop_;
};

std::unique_ptr<BusyWorker> StartCublasWorker(int gpu_id, std::string* error) {
  auto stop = std::make_shared<std::atomic<bool>>(false);
  std::promise<InitResult> init_promise;
  std::future<InitResult> init_future = init_promise.get_future();

  std::thread worker([gpu_id, stop, promise = std::move(init_promise)]() mutable {
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;

    auto cleanup = [&]() {
      if (a != nullptr) cudaFree(a);
      if (b != nullptr) cudaFree(b);
      if (c != nullptr) cudaFree(c);
      if (handle != nullptr) cublasDestroy(handle);
      if (stream != nullptr) cudaStreamDestroy(stream);
    };

    auto fail = [&](const std::string& msg) {
      cleanup();
      promise.set_value(InitResult{false, msg});
    };

    cudaError_t cuda_status = cudaSetDevice(gpu_id);
    if (cuda_status != cudaSuccess) {
      fail("cudaSetDevice(" + std::to_string(gpu_id) + "): " +
           CudaErrorString(cuda_status));
      return;
    }

    cuda_status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (cuda_status != cudaSuccess) {
      fail("cudaStreamCreateWithFlags: " + CudaErrorString(cuda_status));
      return;
    }

    cublasStatus_t cublas_status = cublasCreate(&handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      fail("cublasCreate: " + CublasErrorString(cublas_status));
      return;
    }
    cublas_status = cublasSetStream(handle, stream);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      fail("cublasSetStream: " + CublasErrorString(cublas_status));
      return;
    }

    constexpr int n = 4096;
    constexpr size_t bytes = static_cast<size_t>(n) * n * sizeof(float);
    cuda_status = cudaMalloc(&a, bytes);
    if (cuda_status != cudaSuccess) {
      fail("cudaMalloc A: " + CudaErrorString(cuda_status));
      return;
    }
    cuda_status = cudaMalloc(&b, bytes);
    if (cuda_status != cudaSuccess) {
      fail("cudaMalloc B: " + CudaErrorString(cuda_status));
      return;
    }
    cuda_status = cudaMalloc(&c, bytes);
    if (cuda_status != cudaSuccess) {
      fail("cudaMalloc C: " + CudaErrorString(cuda_status));
      return;
    }

    cudaMemsetAsync(a, 1, bytes, stream);
    cudaMemsetAsync(b, 2, bytes, stream);
    cudaMemsetAsync(c, 0, bytes, stream);
    cuda_status = cudaStreamSynchronize(stream);
    if (cuda_status != cudaSuccess) {
      fail("initial cudaStreamSynchronize: " + CudaErrorString(cuda_status));
      return;
    }

    promise.set_value(InitResult{true, ""});

    const float alpha = 1.0f;
    const float beta = 0.0f;
    while (!stop->load(std::memory_order_relaxed)) {
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
  });

  InitResult init = init_future.get();
  if (!init.ok) {
    stop->store(true, std::memory_order_relaxed);
    if (worker.joinable()) worker.join();
    *error = init.error;
    return nullptr;
  }

  return std::make_unique<CublasBusyWorker>(std::move(worker), std::move(stop));
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
