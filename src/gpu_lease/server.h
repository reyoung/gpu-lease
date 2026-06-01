#pragma once

#include "gpu_lease/busy.h"
#include "gpu_lease/common.h"
#include "gpu_lease/manager.h"

#include <atomic>
#include <functional>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace gpu_lease {

struct ServerOptions {
  std::vector<int> available_ids;
  bool busy_matmul = false;
  BusyBackend* busy_backend = nullptr;
  BusyController::Logger logger;
};

class Server {
 public:
  explicit Server(ServerOptions options);
  ~Server();

  Server(const Server&) = delete;
  Server& operator=(const Server&) = delete;

  bool Run(const std::string& socket_path, std::string* error,
           bool handle_signals = false);
  void Stop();

 private:
  struct Connection {
    int fd = -1;
    std::string input;
    std::string output;
    size_t write_offset = 0;
    bool request_done = false;
    bool close_after_write = false;
    bool lease_active = false;
    bool waiting = false;
    std::string lease_id;
    std::vector<int> lease_ids;
    Request waiting_request;
  };

  bool SetupSocket(const std::string& socket_path, std::string* error);
  bool SetupEpoll(std::string* error, bool handle_signals);
  void CleanupSocket();
  void AcceptConnections();
  void HandleConnectionEvent(int fd, uint32_t events);
  bool ReadConnection(Connection& conn);
  void WriteConnection(Connection& conn);
  void HandleRequest(Connection& conn, const Request& request);
  void QueueResponse(Connection& conn, Response response, bool close_after);
  void CloseConnection(int fd);
  void UpdateConnectionEvents(Connection& conn);
  void ProcessWaitQueue();
  bool TryAcquire(const Request& request, std::string* lease_id,
                  std::vector<int>* ids, std::string* error,
                  bool* waitable);
  void RemoveFromWaitQueue(int fd);

  LeaseManager manager_;
  std::unique_ptr<CudaBusyBackend> owned_busy_backend_;
  std::unique_ptr<BusyController> busy_;
  std::atomic<bool> stopping_{false};
  std::atomic<int> wake_fd_{-1};
  int epoll_fd_ = -1;
  int listen_fd_ = -1;
  int signal_fd_ = -1;
  std::string socket_path_;
  std::unordered_map<int, std::unique_ptr<Connection>> connections_;
  std::list<int> wait_queue_;
};

}  // namespace gpu_lease
