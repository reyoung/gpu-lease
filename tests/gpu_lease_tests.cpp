#include "gpu_lease/app.h"
#include "gpu_lease/busy.h"
#include "gpu_lease/client.h"
#include "gpu_lease/common.h"
#include "gpu_lease/manager.h"
#include "gpu_lease/server.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace gpu_lease;

#define CHECK_TRUE(expr)                                                     \
  do {                                                                       \
    if (!(expr)) {                                                           \
      std::ostringstream _check_stream;                                      \
      _check_stream << __FILE__ << ":" << __LINE__ << ": check failed: "    \
                    << #expr;                                                \
      throw std::runtime_error(_check_stream.str());                         \
    }                                                                        \
  } while (false)

#define CHECK_EQ(got, want)                                                  \
  do {                                                                       \
    const auto _got = (got);                                                  \
    const auto _want = (want);                                                \
    if (!(_got == _want)) {                                                   \
      std::ostringstream _check_stream;                                      \
      _check_stream << __FILE__ << ":" << __LINE__ << ": got " << _got      \
                    << ", want " << _want;                                  \
      throw std::runtime_error(_check_stream.str());                         \
    }                                                                        \
  } while (false)

std::string TempDir() {
  std::string pattern = "/tmp/gpu-lease-test.XXXXXX";
  std::vector<char> writable(pattern.begin(), pattern.end());
  writable.push_back('\0');
  char* path = mkdtemp(writable.data());
  if (path == nullptr) {
    throw std::runtime_error("mkdtemp failed");
  }
  return path;
}

void WaitFor(const std::function<bool()>& predicate, const std::string& label) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  throw std::runtime_error("timed out waiting for " + label);
}

void WaitForSocket(const std::string& socket_path) {
  WaitFor(
      [&]() {
        StatusResult status = GetStatus(socket_path);
        return status.error.empty();
      },
      "socket " + socket_path);
}

class ServerRunner {
 public:
  ServerRunner(ServerOptions options, std::string socket_path)
      : server_(std::move(options)), socket_path_(std::move(socket_path)) {
    thread_ = std::thread([this]() { ok_ = server_.Run(socket_path_, &error_); });
    WaitForSocket(socket_path_);
  }

  ~ServerRunner() {
    if (!stopped_) {
      StopNoThrow();
      if (!ok_ || !error_.empty()) {
        std::cerr << "server stopped with error: " << error_ << "\n";
      }
    }
  }

  void Stop() {
    StopNoThrow();
    CHECK_TRUE(ok_);
    CHECK_TRUE(error_.empty());
  }

 private:
  void StopNoThrow() {
    if (stopped_) return;
    server_.Stop();
    if (thread_.joinable()) thread_.join();
    stopped_ = true;
  }

  Server server_;
  std::string socket_path_;
  std::thread thread_;
  bool ok_ = false;
  bool stopped_ = false;
  std::string error_;
};

class FakeBackend;

class FakeWorker final : public BusyWorker {
 public:
  FakeWorker(FakeBackend* backend, int id) : backend_(backend), id_(id) {}
  void Stop() override;

 private:
  FakeBackend* backend_;
  int id_;
  bool stopped_ = false;
};

class FakeBackend final : public BusyBackend {
 public:
  std::unique_ptr<BusyWorker> Start(int gpu_id, std::string*) override {
    std::lock_guard<std::mutex> lock(mu_);
    starts_.push_back(gpu_id);
    return std::make_unique<FakeWorker>(this, gpu_id);
  }

  void RecordStop(int gpu_id) {
    std::lock_guard<std::mutex> lock(mu_);
    stops_.push_back(gpu_id);
  }

  int Starts(int gpu_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<int>(std::count(starts_.begin(), starts_.end(), gpu_id));
  }

  int Stops(int gpu_id) const {
    std::lock_guard<std::mutex> lock(mu_);
    return static_cast<int>(std::count(stops_.begin(), stops_.end(), gpu_id));
  }

 private:
  mutable std::mutex mu_;
  std::vector<int> starts_;
  std::vector<int> stops_;
};

void FakeWorker::Stop() {
  if (stopped_) return;
  stopped_ = true;
  backend_->RecordStop(id_);
}

void TestParseIDsAndSocketPath() {
  std::string error;
  auto ids = ParseIDs("0, 2,3", &error);
  CHECK_TRUE(ids.has_value());
  CHECK_EQ(IDsEnv(*ids), "0,2,3");

  for (const std::string& input : {"", "0,,1", "-1", "abc", "1,1"}) {
    error.clear();
    CHECK_TRUE(!ParseIDs(input, &error).has_value());
  }

  const char* old = getenv("GPU_LEASE_SOCKET");
  std::string old_value = old == nullptr ? "" : old;
  setenv("GPU_LEASE_SOCKET", "/tmp/env.sock", 1);
  CHECK_EQ(SocketPath(""), "/tmp/env.sock");
  CHECK_EQ(SocketPath("/tmp/flag.sock"), "/tmp/flag.sock");
  if (old == nullptr) {
    unsetenv("GPU_LEASE_SOCKET");
  } else {
    setenv("GPU_LEASE_SOCKET", old_value.c_str(), 1);
  }
}

void TestManagerAndStatusJson() {
  LeaseManager manager({2, 0, 1});
  LeaseSelection selected = manager.SelectAny(2);
  CHECK_TRUE(selected.ok);
  CHECK_EQ(IDsEnv(selected.ids), "0,1");
  const std::string first = manager.Assign(selected.ids);

  LeaseSelection conflict = manager.SelectIDs({1, 2});
  CHECK_TRUE(!conflict.ok);
  CHECK_TRUE(conflict.waitable);

  auto status = manager.Status();
  CHECK_EQ(status["0"], first);
  CHECK_EQ(status["1"], first);

  Response response;
  response.ok = true;
  response.leases = status;
  std::string decode_error;
  auto decoded = DecodeResponse(EncodeResponse(response), &decode_error);
  CHECK_TRUE(decoded.has_value());
  CHECK_EQ(decoded->leases["0"], first);

  manager.Release(first);
  CHECK_TRUE(manager.Status().empty());
}

void TestServerLeaseLifecycle() {
  const std::string socket_path = TempDir() + "/gpu-lease.sock";
  ServerOptions options;
  options.available_ids = {0, 1};
  ServerRunner runner(std::move(options), socket_path);

  AcquireResult first = Acquire(socket_path, {0});
  CHECK_TRUE(first.lease != nullptr);
  AcquireResult conflict = Acquire(socket_path, {0});
  CHECK_TRUE(conflict.lease == nullptr);

  StatusResult status = GetStatus(socket_path);
  CHECK_TRUE(status.error.empty());
  CHECK_EQ(status.leases["0"], first.lease->id());

  first.lease->Close();
  WaitFor(
      [&]() {
        StatusResult empty = GetStatus(socket_path);
        return empty.error.empty() && empty.leases.empty();
      },
      "lease release");
  runner.Stop();
}

void TestServerWaitQueue() {
  const std::string socket_path = TempDir() + "/gpu-lease.sock";
  ServerOptions options;
  options.available_ids = {0};
  ServerRunner runner(std::move(options), socket_path);

  AcquireResult first = AcquireWithOptions(socket_path, AcquireOptions{{}, 1, false});
  CHECK_TRUE(first.lease != nullptr);

  std::atomic<bool> done{false};
  AcquireResult waited;
  std::thread waiter([&]() {
    waited = AcquireWithOptions(socket_path, AcquireOptions{{}, 1, true});
    done.store(true);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  CHECK_TRUE(!done.load());
  first.lease->Close();
  WaitFor([&]() { return done.load(); }, "waiting acquire");
  waiter.join();
  CHECK_TRUE(waited.lease != nullptr);
  CHECK_EQ(IDsEnv(waited.lease->ids()), "0");
  waited.lease->Close();
  runner.Stop();
}

void TestRunCommandSetsCUDAVisibleDevicesAndReleasesLease() {
  const std::string socket_path = TempDir() + "/gpu-lease.sock";
  ServerOptions options;
  options.available_ids = {0, 1};
  ServerRunner runner(std::move(options), socket_path);

  std::ostringstream out;
  std::ostringstream err;
  const int code = RunCLI({"run", "--socket", socket_path, "--ids", "0,1", "--",
                           "/bin/sh", "-c",
                           "test \"$CUDA_VISIBLE_DEVICES\" = \"0,1\""},
                          out, err);
  CHECK_EQ(code, 0);
  WaitFor(
      [&]() {
        StatusResult status = GetStatus(socket_path);
        return status.error.empty() && status.leases.empty();
      },
      "run lease release");
  runner.Stop();
}

void TestRunCommandWaitsForCount() {
  const std::string socket_path = TempDir() + "/gpu-lease.sock";
  ServerOptions options;
  options.available_ids = {0};
  ServerRunner runner(std::move(options), socket_path);

  AcquireResult first = AcquireWithOptions(socket_path, AcquireOptions{{}, 1, false});
  CHECK_TRUE(first.lease != nullptr);

  std::ostringstream out;
  std::ostringstream err;
  std::atomic<bool> done{false};
  int code = -1;
  std::thread run_thread([&]() {
    code = RunCLI({"run", "--socket", socket_path, "--count", "1", "--wait",
                   "--", "/bin/sh", "-c",
                   "test \"$CUDA_VISIBLE_DEVICES\" = \"0\""},
                  out, err);
    done.store(true);
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  CHECK_TRUE(!done.load());
  first.lease->Close();
  WaitFor([&]() { return done.load(); }, "waiting run command");
  run_thread.join();
  CHECK_EQ(code, 0);
  runner.Stop();
}

void TestSocketModeIgnoresRestrictiveUmask() {
  struct UmaskGuard {
    explicit UmaskGuard(mode_t new_umask) : old(umask(new_umask)) {}
    ~UmaskGuard() { umask(old); }
    mode_t old;
  } guard(0077);
  const std::string socket_path = TempDir() + "/gpu-lease.sock";
  ServerOptions options;
  options.available_ids = {0};
  {
    ServerRunner runner(std::move(options), socket_path);
    struct stat st;
    CHECK_EQ(stat(socket_path.c_str(), &st), 0);
    CHECK_EQ(static_cast<int>(st.st_mode & 0777), 0666);
    runner.Stop();
  }
}

void TestBusyWorkerLifecycleWithFakeBackend() {
  const std::string socket_path = TempDir() + "/gpu-lease.sock";
  FakeBackend backend;
  ServerOptions options;
  options.available_ids = {0};
  options.busy_matmul = true;
  options.busy_backend = &backend;
  ServerRunner runner(std::move(options), socket_path);

  WaitFor([&]() { return backend.Starts(0) == 1; }, "initial busy start");
  AcquireResult held = AcquireWithOptions(socket_path, AcquireOptions{{}, 1, false});
  CHECK_TRUE(held.lease != nullptr);
  WaitFor([&]() { return backend.Stops(0) == 1; }, "busy stop before lease");
  CHECK_EQ(backend.Starts(0), 1);

  held.lease->Close();
  WaitFor([&]() { return backend.Starts(0) == 2; }, "busy restart");
  runner.Stop();
}

using TestFn = void (*)();

void RunTest(const std::string& name, TestFn fn) {
  fn();
  std::cout << "ok " << name << "\n";
}

}  // namespace

int main() {
  setenv("GPU_LEASE_DISABLE_PRESTART_CHECK", "1", 1);
  try {
    RunTest("parse ids and socket path", TestParseIDsAndSocketPath);
    RunTest("manager and status json", TestManagerAndStatusJson);
    RunTest("server lease lifecycle", TestServerLeaseLifecycle);
    RunTest("server wait queue", TestServerWaitQueue);
    RunTest("run command env and release",
            TestRunCommandSetsCUDAVisibleDevicesAndReleasesLease);
    RunTest("run command waits for count", TestRunCommandWaitsForCount);
    RunTest("socket mode", TestSocketModeIgnoresRestrictiveUmask);
    RunTest("busy worker fake backend", TestBusyWorkerLifecycleWithFakeBackend);
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << "\n";
    return 1;
  }
  return 0;
}
