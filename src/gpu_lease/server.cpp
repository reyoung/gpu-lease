#include "gpu_lease/server.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stddef.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/signalfd.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <iostream>
#include <utility>

namespace gpu_lease {
namespace {

bool SetNonBlocking(int fd, std::string* error) {
  const int flags = fcntl(fd, F_GETFL, 0);
  if (flags < 0) {
    *error = ErrnoString(errno);
    return false;
  }
  if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0) {
    *error = ErrnoString(errno);
    return false;
  }
  return true;
}

std::string ParentDir(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  if (slash == std::string::npos) return ".";
  if (slash == 0) return "/";
  return path.substr(0, slash);
}

bool MakeDirs(const std::string& path, std::string* error) {
  if (path.empty() || path == ".") return true;
  std::string current;
  size_t pos = 0;
  if (path[0] == '/') {
    current = "/";
    pos = 1;
  }
  while (pos <= path.size()) {
    const size_t slash = path.find('/', pos);
    const std::string part =
        path.substr(pos, slash == std::string::npos ? std::string::npos
                                                    : slash - pos);
    if (!part.empty()) {
      if (current.empty() || current == "/") {
        current += part;
      } else {
        current += "/" + part;
      }
      if (mkdir(current.c_str(), 0755) != 0 && errno != EEXIST) {
        *error = current + ": " + ErrnoString(errno);
        return false;
      }
    }
    if (slash == std::string::npos) break;
    pos = slash + 1;
  }
  return true;
}

bool AddEpoll(int epoll_fd, int fd, uint32_t events, std::string* error) {
  epoll_event event;
  memset(&event, 0, sizeof(event));
  event.events = events;
  event.data.fd = fd;
  if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &event) != 0) {
    *error = ErrnoString(errno);
    return false;
  }
  return true;
}

bool ModEpoll(int epoll_fd, int fd, uint32_t events, std::string* error) {
  epoll_event event;
  memset(&event, 0, sizeof(event));
  event.events = events;
  event.data.fd = fd;
  if (epoll_ctl(epoll_fd, EPOLL_CTL_MOD, fd, &event) != 0) {
    *error = ErrnoString(errno);
    return false;
  }
  return true;
}

}  // namespace

Server::Server(ServerOptions options)
    : manager_(std::move(options.available_ids)) {
  if (options.busy_matmul) {
    BusyBackend* backend = options.busy_backend;
    if (backend == nullptr) {
      owned_busy_backend_ = std::make_unique<CudaBusyBackend>();
      backend = owned_busy_backend_.get();
    }
    BusyController::Logger logger = std::move(options.logger);
    if (!logger) {
      logger = [](const std::string& line) { std::cerr << line << "\n"; };
    }
    busy_ = std::make_unique<BusyController>(manager_.AvailableIDs(), backend,
                                             std::move(logger));
  }
}

Server::~Server() {
  Stop();
  if (busy_) busy_->StopAll();
}

bool Server::Run(const std::string& socket_path, std::string* error,
                 bool handle_signals) {
  if (socket_path.empty()) {
    *error = "socket path must not be empty";
    return false;
  }
  stopping_.store(false, std::memory_order_relaxed);
  socket_path_ = socket_path;

  if (!SetupSocket(socket_path, error)) {
    CleanupSocket();
    return false;
  }
  if (!SetupEpoll(error, handle_signals)) {
    CleanupSocket();
    return false;
  }

  if (busy_) {
    busy_->StartIdle(manager_);
  }

  constexpr int kMaxEvents = 64;
  epoll_event events[kMaxEvents];
  while (!stopping_.load(std::memory_order_relaxed)) {
    const int n = epoll_wait(epoll_fd_, events, kMaxEvents, -1);
    if (n < 0) {
      if (errno == EINTR) continue;
      *error = ErrnoString(errno);
      CleanupSocket();
      return false;
    }
    for (int i = 0; i < n; ++i) {
      const int fd = events[i].data.fd;
      if (fd == listen_fd_) {
        AcceptConnections();
      } else if (fd == wake_fd_.load(std::memory_order_relaxed)) {
        uint64_t value = 0;
        while (read(fd, &value, sizeof(value)) > 0) {
        }
        stopping_.store(true, std::memory_order_relaxed);
      } else if (signal_fd_ >= 0 && fd == signal_fd_) {
        signalfd_siginfo info;
        while (read(signal_fd_, &info, sizeof(info)) > 0) {
        }
        stopping_.store(true, std::memory_order_relaxed);
      } else {
        HandleConnectionEvent(fd, events[i].events);
      }
    }
  }

  CleanupSocket();
  return true;
}

void Server::Stop() {
  stopping_.store(true, std::memory_order_relaxed);
  const int fd = wake_fd_.load(std::memory_order_relaxed);
  if (fd >= 0) {
    uint64_t one = 1;
    (void)write(fd, &one, sizeof(one));
  }
}

bool Server::SetupSocket(const std::string& socket_path, std::string* error) {
  if (!MakeDirs(ParentDir(socket_path), error)) {
    return false;
  }

  if (unlink(socket_path.c_str()) != 0 && errno != ENOENT) {
    *error = socket_path + ": " + ErrnoString(errno);
    return false;
  }

  listen_fd_ = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
  if (listen_fd_ < 0) {
    *error = ErrnoString(errno);
    return false;
  }

  sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  if (socket_path.size() >= sizeof(addr.sun_path)) {
    *error = "socket path is too long";
    return false;
  }
  memcpy(addr.sun_path, socket_path.c_str(), socket_path.size() + 1);
  const socklen_t len =
      static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) +
                             socket_path.size() + 1);
  if (bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), len) != 0) {
    *error = ErrnoString(errno);
    return false;
  }
  if (listen(listen_fd_, SOMAXCONN) != 0) {
    *error = ErrnoString(errno);
    return false;
  }
  if (chmod(socket_path.c_str(), 0666) != 0) {
    *error = ErrnoString(errno);
    return false;
  }
  return true;
}

bool Server::SetupEpoll(std::string* error, bool handle_signals) {
  epoll_fd_ = epoll_create1(EPOLL_CLOEXEC);
  if (epoll_fd_ < 0) {
    *error = ErrnoString(errno);
    return false;
  }
  if (!AddEpoll(epoll_fd_, listen_fd_, EPOLLIN, error)) {
    return false;
  }

  const int event_fd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  if (event_fd < 0) {
    *error = ErrnoString(errno);
    return false;
  }
  wake_fd_.store(event_fd, std::memory_order_relaxed);
  if (!AddEpoll(epoll_fd_, event_fd, EPOLLIN, error)) {
    return false;
  }

  if (handle_signals) {
    sigset_t mask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGINT);
    sigaddset(&mask, SIGTERM);
    if (sigprocmask(SIG_BLOCK, &mask, nullptr) != 0) {
      *error = ErrnoString(errno);
      return false;
    }
    signal_fd_ = signalfd(-1, &mask, SFD_NONBLOCK | SFD_CLOEXEC);
    if (signal_fd_ < 0) {
      *error = ErrnoString(errno);
      return false;
    }
    if (!AddEpoll(epoll_fd_, signal_fd_, EPOLLIN, error)) {
      return false;
    }
  }

  return true;
}

void Server::CleanupSocket() {
  if (busy_) {
    busy_->StopAll();
  }
  for (auto& [_, conn] : connections_) {
    if (conn->fd >= 0) close(conn->fd);
  }
  connections_.clear();
  wait_queue_.clear();

  if (signal_fd_ >= 0) {
    close(signal_fd_);
    signal_fd_ = -1;
  }
  const int event_fd = wake_fd_.exchange(-1, std::memory_order_relaxed);
  if (event_fd >= 0) close(event_fd);
  if (epoll_fd_ >= 0) {
    close(epoll_fd_);
    epoll_fd_ = -1;
  }
  if (listen_fd_ >= 0) {
    close(listen_fd_);
    listen_fd_ = -1;
  }
  if (!socket_path_.empty()) {
    unlink(socket_path_.c_str());
    socket_path_.clear();
  }
}

void Server::AcceptConnections() {
  while (true) {
    const int fd = accept4(listen_fd_, nullptr, nullptr,
                           SOCK_NONBLOCK | SOCK_CLOEXEC);
    if (fd < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) return;
      return;
    }
    std::string error;
    if (!SetNonBlocking(fd, &error) ||
        !AddEpoll(epoll_fd_, fd, EPOLLIN | EPOLLRDHUP, &error)) {
      close(fd);
      continue;
    }
    auto conn = std::make_unique<Connection>();
    conn->fd = fd;
    connections_[fd] = std::move(conn);
  }
}

void Server::HandleConnectionEvent(int fd, uint32_t events) {
  auto it = connections_.find(fd);
  if (it == connections_.end()) return;
  Connection* conn = it->second.get();

  if (events & EPOLLIN) {
    if (!ReadConnection(*conn)) {
      CloseConnection(fd);
      return;
    }
  }

  it = connections_.find(fd);
  if (it == connections_.end()) return;
  conn = it->second.get();

  if (events & EPOLLOUT) {
    WriteConnection(*conn);
  }

  it = connections_.find(fd);
  if (it == connections_.end()) return;
  conn = it->second.get();

  if ((events & (EPOLLHUP | EPOLLERR)) ||
      ((events & EPOLLRDHUP) && conn->output.empty())) {
    CloseConnection(fd);
  }
}

bool Server::ReadConnection(Connection& conn) {
  char buffer[4096];
  while (true) {
    const ssize_t n = read(conn.fd, buffer, sizeof(buffer));
    if (n > 0) {
      if (!conn.request_done) {
        conn.input.append(buffer, static_cast<size_t>(n));
        const size_t newline = conn.input.find('\n');
        if (newline != std::string::npos) {
          const std::string line = conn.input.substr(0, newline);
          conn.request_done = true;
          std::string error;
          auto request = DecodeRequest(line, &error);
          if (!request) {
            QueueResponse(conn,
                          Response{false, "decode request: " + error, "", {},
                                   {}},
                          true);
          } else {
            HandleRequest(conn, *request);
          }
        }
      }
      continue;
    }
    if (n == 0) {
      return false;
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return true;
    }
    if (errno == EINTR) {
      continue;
    }
    return false;
  }
}

void Server::WriteConnection(Connection& conn) {
  while (conn.write_offset < conn.output.size()) {
    const ssize_t n =
        write(conn.fd, conn.output.data() + conn.write_offset,
              conn.output.size() - conn.write_offset);
    if (n > 0) {
      conn.write_offset += static_cast<size_t>(n);
      continue;
    }
    if (n < 0 && errno == EINTR) continue;
    if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      return;
    }
    CloseConnection(conn.fd);
    return;
  }

  conn.output.clear();
  conn.write_offset = 0;
  if (conn.close_after_write) {
    CloseConnection(conn.fd);
    return;
  }
  UpdateConnectionEvents(conn);
}

void Server::HandleRequest(Connection& conn, const Request& request) {
  if (request.action == kActionStatus) {
    Response response;
    response.ok = true;
    response.leases = manager_.Status();
    QueueResponse(conn, std::move(response), true);
    return;
  }

  if (request.action != kActionLease) {
    QueueResponse(conn, Response{false, "unknown action", "", {}, {}}, true);
    return;
  }

  std::string lease_id;
  std::vector<int> ids;
  std::string error;
  bool waitable = false;
  if (TryAcquire(request, &lease_id, &ids, &error, &waitable)) {
    conn.lease_active = true;
    conn.lease_id = lease_id;
    conn.lease_ids = ids;
    Response response;
    response.ok = true;
    response.lease = lease_id;
    response.ids = ids;
    QueueResponse(conn, std::move(response), false);
    return;
  }

  if (request.wait && waitable) {
    conn.waiting = true;
    conn.waiting_request = request;
    wait_queue_.push_back(conn.fd);
    UpdateConnectionEvents(conn);
    return;
  }

  QueueResponse(conn, Response{false, error, "", {}, {}}, true);
}

void Server::QueueResponse(Connection& conn, Response response,
                           bool close_after) {
  conn.output += EncodeResponse(response);
  conn.close_after_write = close_after;
  UpdateConnectionEvents(conn);
}

void Server::CloseConnection(int fd) {
  auto it = connections_.find(fd);
  if (it == connections_.end()) return;

  std::vector<int> released;
  const bool had_lease = it->second->lease_active;
  if (it->second->waiting) {
    RemoveFromWaitQueue(fd);
  }
  if (had_lease) {
    released = manager_.Release(it->second->lease_id);
  }

  epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr);
  close(fd);
  connections_.erase(it);

  if (had_lease) {
    ProcessWaitQueue();
    if (busy_) {
      busy_->RestartReleased(released, manager_);
    }
  }
}

void Server::UpdateConnectionEvents(Connection& conn) {
  if (epoll_fd_ < 0 || conn.fd < 0) return;
  uint32_t events = EPOLLIN | EPOLLRDHUP;
  if (!conn.output.empty()) events |= EPOLLOUT;
  std::string error;
  (void)ModEpoll(epoll_fd_, conn.fd, events, &error);
}

void Server::ProcessWaitQueue() {
  for (auto it = wait_queue_.begin(); it != wait_queue_.end();) {
    const int fd = *it;
    auto conn_it = connections_.find(fd);
    if (conn_it == connections_.end() || !conn_it->second->waiting) {
      it = wait_queue_.erase(it);
      continue;
    }

    Connection& conn = *conn_it->second;
    std::string lease_id;
    std::vector<int> ids;
    std::string error;
    bool waitable = false;
    if (!TryAcquire(conn.waiting_request, &lease_id, &ids, &error, &waitable)) {
      if (!waitable) {
        conn.waiting = false;
        it = wait_queue_.erase(it);
        QueueResponse(conn, Response{false, error, "", {}, {}}, true);
      } else {
        ++it;
      }
      continue;
    }

    conn.waiting = false;
    conn.lease_active = true;
    conn.lease_id = lease_id;
    conn.lease_ids = ids;
    it = wait_queue_.erase(it);

    Response response;
    response.ok = true;
    response.lease = lease_id;
    response.ids = ids;
    QueueResponse(conn, std::move(response), false);
  }
}

bool Server::TryAcquire(const Request& request, std::string* lease_id,
                        std::vector<int>* ids, std::string* error,
                        bool* waitable) {
  *waitable = false;
  if (!request.ids.empty() && request.count > 0) {
    *error = "ids and count cannot both be set";
    return false;
  }

  LeaseSelection selection =
      request.count > 0 ? manager_.SelectAny(request.count)
                        : manager_.SelectIDs(request.ids);
  if (!selection.ok) {
    *waitable = selection.waitable;
    *error = selection.error;
    return false;
  }

  if (busy_) {
    busy_->StopForLease(selection.ids);
  }
  *lease_id = manager_.Assign(selection.ids);
  *ids = std::move(selection.ids);
  return true;
}

void Server::RemoveFromWaitQueue(int fd) {
  wait_queue_.remove(fd);
}

}  // namespace gpu_lease
