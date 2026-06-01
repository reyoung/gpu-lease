#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace gpu_lease {

constexpr const char* kDefaultSocketPath = "/var/run/gpu-lease.sock";
constexpr const char* kActionLease = "lease";
constexpr const char* kActionStatus = "status";

struct Request {
  std::string action;
  std::vector<int> ids;
  int count = 0;
  bool wait = false;
};

struct Response {
  bool ok = false;
  std::string error;
  std::string lease;
  std::vector<int> ids;
  std::map<std::string, std::string> leases;
};

std::string SocketPath(const std::string& flag_value);
std::optional<std::vector<int>> ParseIDs(const std::string& raw,
                                         std::string* error);
std::string IDsEnv(const std::vector<int>& ids);
std::vector<int> NormalizeIDs(const std::vector<int>& ids);
std::vector<int> DiscoverGPUIDs();

std::string EncodeRequest(const Request& request);
std::string EncodeResponse(const Response& response);
std::optional<Request> DecodeRequest(const std::string& json,
                                     std::string* error);
std::optional<Response> DecodeResponse(const std::string& json,
                                       std::string* error);

std::string ErrnoString(int err);
bool SendAll(int fd, const std::string& data, std::string* error);
std::optional<std::string> ReadJsonLine(int fd, std::string* error);

}  // namespace gpu_lease
