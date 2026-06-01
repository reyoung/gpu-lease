#include "gpu_lease/client.h"

#include "gpu_lease/common.h"

#include <errno.h>
#include <stddef.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <utility>

namespace gpu_lease {
namespace {

int ConnectUnixSocket(const std::string& socket_path, std::string* error) {
  int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (fd < 0) {
    *error = ErrnoString(errno);
    return -1;
  }

  sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  if (socket_path.size() >= sizeof(addr.sun_path)) {
    *error = "socket path is too long";
    close(fd);
    return -1;
  }
  memcpy(addr.sun_path, socket_path.c_str(), socket_path.size() + 1);
  const socklen_t len =
      static_cast<socklen_t>(offsetof(sockaddr_un, sun_path) +
                             socket_path.size() + 1);
  if (connect(fd, reinterpret_cast<sockaddr*>(&addr), len) != 0) {
    *error = ErrnoString(errno);
    close(fd);
    return -1;
  }
  return fd;
}

}  // namespace

HeldLease::HeldLease(std::string id, std::vector<int> ids, int fd)
    : id_(std::move(id)), ids_(std::move(ids)), fd_(fd) {}

HeldLease::~HeldLease() { Close(); }

HeldLease::HeldLease(HeldLease&& other) noexcept
    : id_(std::move(other.id_)), ids_(std::move(other.ids_)), fd_(other.fd_) {
  other.fd_ = -1;
}

HeldLease& HeldLease::operator=(HeldLease&& other) noexcept {
  if (this == &other) return *this;
  Close();
  id_ = std::move(other.id_);
  ids_ = std::move(other.ids_);
  fd_ = other.fd_;
  other.fd_ = -1;
  return *this;
}

void HeldLease::Close() {
  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
  }
}

AcquireResult AcquireWithOptions(const std::string& socket_path,
                                 const AcquireOptions& options) {
  AcquireResult result;
  std::string error;
  const int fd = ConnectUnixSocket(socket_path, &error);
  if (fd < 0) {
    result.error = error;
    return result;
  }

  Request request;
  request.action = kActionLease;
  request.ids = options.ids;
  request.count = options.count;
  request.wait = options.wait;
  if (!SendAll(fd, EncodeRequest(request), &error)) {
    close(fd);
    result.error = error;
    return result;
  }

  auto line = ReadJsonLine(fd, &error);
  if (!line) {
    close(fd);
    result.error = error;
    return result;
  }
  auto response = DecodeResponse(*line, &error);
  if (!response) {
    close(fd);
    result.error = "decode response: " + error;
    return result;
  }
  if (!response->ok) {
    close(fd);
    result.error = response->error;
    return result;
  }

  result.lease = std::make_unique<HeldLease>(response->lease, response->ids, fd);
  return result;
}

AcquireResult Acquire(const std::string& socket_path,
                      const std::vector<int>& ids) {
  AcquireOptions options;
  options.ids = ids;
  return AcquireWithOptions(socket_path, options);
}

StatusResult GetStatus(const std::string& socket_path) {
  StatusResult result;
  std::string error;
  const int fd = ConnectUnixSocket(socket_path, &error);
  if (fd < 0) {
    result.error = error;
    return result;
  }

  Request request;
  request.action = kActionStatus;
  if (!SendAll(fd, EncodeRequest(request), &error)) {
    close(fd);
    result.error = error;
    return result;
  }

  auto line = ReadJsonLine(fd, &error);
  close(fd);
  if (!line) {
    result.error = error;
    return result;
  }
  auto response = DecodeResponse(*line, &error);
  if (!response) {
    result.error = "decode response: " + error;
    return result;
  }
  if (!response->ok) {
    result.error = response->error;
    return result;
  }
  result.leases = std::move(response->leases);
  return result;
}

}  // namespace gpu_lease
