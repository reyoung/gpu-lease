#include "gpu_lease/common.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <map>
#include <set>
#include <sstream>
#include <string_view>
#include <utility>

namespace gpu_lease {
namespace {

struct JsonValue {
  enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };

  Type type = Type::kNull;
  bool boolean = false;
  long long number = 0;
  std::string string;
  std::vector<JsonValue> array;
  std::map<std::string, JsonValue> object;
};

class JsonParser {
 public:
  explicit JsonParser(std::string_view input) : input_(input) {}

  std::optional<JsonValue> Parse(std::string* error) {
    SkipSpace();
    auto value = ParseValue(error);
    if (!value) {
      return std::nullopt;
    }
    SkipSpace();
    if (pos_ != input_.size()) {
      *error = "trailing data after JSON value";
      return std::nullopt;
    }
    return value;
  }

 private:
  std::optional<JsonValue> ParseValue(std::string* error) {
    SkipSpace();
    if (pos_ >= input_.size()) {
      *error = "unexpected end of JSON";
      return std::nullopt;
    }
    const char c = input_[pos_];
    if (c == '"') {
      auto s = ParseString(error);
      if (!s) return std::nullopt;
      JsonValue value;
      value.type = JsonValue::Type::kString;
      value.string = *s;
      return value;
    }
    if (c == '{') return ParseObject(error);
    if (c == '[') return ParseArray(error);
    if (c == 't' || c == 'f') return ParseBool(error);
    if (c == 'n') return ParseNull(error);
    if (c == '-' || isdigit(static_cast<unsigned char>(c))) {
      return ParseNumber(error);
    }
    *error = "unexpected JSON token";
    return std::nullopt;
  }

  std::optional<JsonValue> ParseNull(std::string* error) {
    if (input_.substr(pos_, 4) != "null") {
      *error = "invalid null literal";
      return std::nullopt;
    }
    pos_ += 4;
    JsonValue value;
    value.type = JsonValue::Type::kNull;
    return value;
  }

  std::optional<JsonValue> ParseBool(std::string* error) {
    JsonValue value;
    value.type = JsonValue::Type::kBool;
    if (input_.substr(pos_, 4) == "true") {
      value.boolean = true;
      pos_ += 4;
      return value;
    }
    if (input_.substr(pos_, 5) == "false") {
      value.boolean = false;
      pos_ += 5;
      return value;
    }
    *error = "invalid boolean literal";
    return std::nullopt;
  }

  std::optional<JsonValue> ParseNumber(std::string* error) {
    const size_t start = pos_;
    if (input_[pos_] == '-') ++pos_;
    if (pos_ >= input_.size() ||
        !isdigit(static_cast<unsigned char>(input_[pos_]))) {
      *error = "invalid number";
      return std::nullopt;
    }
    while (pos_ < input_.size() &&
           isdigit(static_cast<unsigned char>(input_[pos_]))) {
      ++pos_;
    }
    if (pos_ < input_.size() && (input_[pos_] == '.' || input_[pos_] == 'e' ||
                                 input_[pos_] == 'E')) {
      *error = "floating point numbers are not accepted";
      return std::nullopt;
    }

    JsonValue value;
    value.type = JsonValue::Type::kNumber;
    try {
      value.number = std::stoll(std::string(input_.substr(start, pos_ - start)));
    } catch (const std::exception&) {
      *error = "invalid number";
      return std::nullopt;
    }
    return value;
  }

  std::optional<std::string> ParseString(std::string* error) {
    if (input_[pos_] != '"') {
      *error = "expected string";
      return std::nullopt;
    }
    ++pos_;
    std::string out;
    while (pos_ < input_.size()) {
      const char c = input_[pos_++];
      if (c == '"') {
        return out;
      }
      if (static_cast<unsigned char>(c) < 0x20) {
        *error = "control character in JSON string";
        return std::nullopt;
      }
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (pos_ >= input_.size()) {
        *error = "unterminated JSON escape";
        return std::nullopt;
      }
      const char esc = input_[pos_++];
      switch (esc) {
        case '"':
        case '\\':
        case '/':
          out.push_back(esc);
          break;
        case 'b':
          out.push_back('\b');
          break;
        case 'f':
          out.push_back('\f');
          break;
        case 'n':
          out.push_back('\n');
          break;
        case 'r':
          out.push_back('\r');
          break;
        case 't':
          out.push_back('\t');
          break;
        case 'u':
          if (pos_ + 4 > input_.size()) {
            *error = "truncated unicode escape";
            return std::nullopt;
          }
          // The protocol only needs ASCII field names and values. Preserve
          // non-ASCII escapes as '?' instead of rejecting otherwise valid JSON.
          pos_ += 4;
          out.push_back('?');
          break;
        default:
          *error = "invalid JSON escape";
          return std::nullopt;
      }
    }
    *error = "unterminated JSON string";
    return std::nullopt;
  }

  std::optional<JsonValue> ParseArray(std::string* error) {
    ++pos_;
    JsonValue value;
    value.type = JsonValue::Type::kArray;
    SkipSpace();
    if (pos_ < input_.size() && input_[pos_] == ']') {
      ++pos_;
      return value;
    }
    while (true) {
      auto item = ParseValue(error);
      if (!item) return std::nullopt;
      value.array.push_back(std::move(*item));
      SkipSpace();
      if (pos_ >= input_.size()) {
        *error = "unterminated JSON array";
        return std::nullopt;
      }
      if (input_[pos_] == ']') {
        ++pos_;
        return value;
      }
      if (input_[pos_] != ',') {
        *error = "expected comma in JSON array";
        return std::nullopt;
      }
      ++pos_;
    }
  }

  std::optional<JsonValue> ParseObject(std::string* error) {
    ++pos_;
    JsonValue value;
    value.type = JsonValue::Type::kObject;
    SkipSpace();
    if (pos_ < input_.size() && input_[pos_] == '}') {
      ++pos_;
      return value;
    }
    while (true) {
      SkipSpace();
      auto key = ParseString(error);
      if (!key) return std::nullopt;
      SkipSpace();
      if (pos_ >= input_.size() || input_[pos_] != ':') {
        *error = "expected colon in JSON object";
        return std::nullopt;
      }
      ++pos_;
      auto item = ParseValue(error);
      if (!item) return std::nullopt;
      value.object[*key] = std::move(*item);
      SkipSpace();
      if (pos_ >= input_.size()) {
        *error = "unterminated JSON object";
        return std::nullopt;
      }
      if (input_[pos_] == '}') {
        ++pos_;
        return value;
      }
      if (input_[pos_] != ',') {
        *error = "expected comma in JSON object";
        return std::nullopt;
      }
      ++pos_;
    }
  }

  void SkipSpace() {
    while (pos_ < input_.size() &&
           isspace(static_cast<unsigned char>(input_[pos_]))) {
      ++pos_;
    }
  }

  std::string_view input_;
  size_t pos_ = 0;
};

std::string JsonEscape(const std::string& input) {
  std::string out;
  out.reserve(input.size() + 8);
  for (const char c : input) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[7];
          snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out.push_back(c);
        }
    }
  }
  return out;
}

std::string EncodeIDs(const std::vector<int>& ids) {
  std::ostringstream out;
  out << '[';
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i != 0) out << ',';
    out << ids[i];
  }
  out << ']';
  return out.str();
}

std::optional<std::vector<int>> DecodeIDs(const JsonValue& value,
                                          std::string* error) {
  if (value.type != JsonValue::Type::kArray) {
    *error = "ids must be an array";
    return std::nullopt;
  }
  std::vector<int> ids;
  ids.reserve(value.array.size());
  for (const JsonValue& item : value.array) {
    if (item.type != JsonValue::Type::kNumber || item.number < 0 ||
        item.number > INT32_MAX) {
      *error = "ids must contain non-negative integers";
      return std::nullopt;
    }
    ids.push_back(static_cast<int>(item.number));
  }
  return ids;
}

std::optional<std::vector<int>> DiscoverGPUIDsWithNvidiaSMI() {
  int pipefd[2];
  if (pipe(pipefd) != 0) {
    return std::nullopt;
  }

  const pid_t pid = fork();
  if (pid < 0) {
    close(pipefd[0]);
    close(pipefd[1]);
    return std::nullopt;
  }

  if (pid == 0) {
    dup2(pipefd[1], STDOUT_FILENO);
    const int devnull = open("/dev/null", O_WRONLY);
    if (devnull >= 0) dup2(devnull, STDERR_FILENO);
    close(pipefd[0]);
    close(pipefd[1]);
    execlp("nvidia-smi", "nvidia-smi", "--query-gpu=index",
           "--format=csv,noheader", static_cast<char*>(nullptr));
    _exit(127);
  }

  close(pipefd[1]);
  fcntl(pipefd[0], F_SETFL, fcntl(pipefd[0], F_GETFL, 0) | O_NONBLOCK);

  std::string output;
  const int deadline_ms = 2000;
  int elapsed_ms = 0;
  int status = 0;
  bool exited = false;
  while (elapsed_ms < deadline_ms) {
    char buffer[512];
    while (true) {
      const ssize_t n = read(pipefd[0], buffer, sizeof(buffer));
      if (n > 0) {
        output.append(buffer, static_cast<size_t>(n));
        continue;
      }
      if (n == 0) break;
      if (errno == EAGAIN || errno == EWOULDBLOCK) break;
      close(pipefd[0]);
      kill(pid, SIGKILL);
      waitpid(pid, nullptr, 0);
      return std::nullopt;
    }

    const pid_t got = waitpid(pid, &status, WNOHANG);
    if (got == pid) {
      exited = true;
      break;
    }
    if (got < 0) {
      close(pipefd[0]);
      return std::nullopt;
    }

    struct pollfd pfd;
    pfd.fd = pipefd[0];
    pfd.events = POLLIN;
    pfd.revents = 0;
    const int step_ms = 50;
    poll(&pfd, 1, step_ms);
    elapsed_ms += step_ms;
  }

  if (!exited) {
    kill(pid, SIGKILL);
    waitpid(pid, nullptr, 0);
    close(pipefd[0]);
    return std::nullopt;
  }

  char buffer[512];
  while (true) {
    const ssize_t n = read(pipefd[0], buffer, sizeof(buffer));
    if (n > 0) {
      output.append(buffer, static_cast<size_t>(n));
      continue;
    }
    break;
  }
  close(pipefd[0]);

  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    return std::nullopt;
  }

  std::vector<int> ids;
  std::istringstream lines(output);
  std::string line;
  while (std::getline(lines, line)) {
    line.erase(line.begin(),
               std::find_if(line.begin(), line.end(), [](unsigned char c) {
                 return !isspace(c);
               }));
    line.erase(std::find_if(line.rbegin(), line.rend(), [](unsigned char c) {
                 return !isspace(c);
               }).base(),
               line.end());
    if (line.empty()) continue;
    char* end = nullptr;
    const long parsed = strtol(line.c_str(), &end, 10);
    if (end != line.c_str() && *end == '\0' && parsed >= 0 &&
        parsed <= INT32_MAX) {
      ids.push_back(static_cast<int>(parsed));
    }
  }
  return NormalizeIDs(ids);
}

std::vector<int> DiscoverGPUIDsFromDev() {
  DIR* dir = opendir("/dev");
  if (dir == nullptr) {
    return {};
  }

  std::vector<int> ids;
  while (dirent* entry = readdir(dir)) {
    const std::string name(entry->d_name);
    constexpr const char* prefix = "nvidia";
    if (name.rfind(prefix, 0) != 0) continue;
    const std::string suffix = name.substr(strlen(prefix));
    if (suffix.empty()) continue;
    if (!std::all_of(suffix.begin(), suffix.end(), [](unsigned char c) {
          return isdigit(c);
        })) {
      continue;
    }
    const long parsed = strtol(suffix.c_str(), nullptr, 10);
    if (parsed >= 0 && parsed <= INT32_MAX) {
      ids.push_back(static_cast<int>(parsed));
    }
  }
  closedir(dir);
  return NormalizeIDs(ids);
}

}  // namespace

std::string SocketPath(const std::string& flag_value) {
  if (!flag_value.empty()) {
    return flag_value;
  }
  const char* env = getenv("GPU_LEASE_SOCKET");
  if (env != nullptr && env[0] != '\0') {
    return env;
  }
  return kDefaultSocketPath;
}

std::optional<std::vector<int>> ParseIDs(const std::string& raw,
                                         std::string* error) {
  const auto first = raw.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    *error = "ids must not be empty";
    return std::nullopt;
  }

  std::vector<int> ids;
  std::set<int> seen;
  size_t start = 0;
  while (start <= raw.size()) {
    const size_t comma = raw.find(',', start);
    const size_t end = comma == std::string::npos ? raw.size() : comma;
    std::string part = raw.substr(start, end - start);
    part.erase(part.begin(),
               std::find_if(part.begin(), part.end(), [](unsigned char c) {
                 return !isspace(c);
               }));
    part.erase(std::find_if(part.rbegin(), part.rend(), [](unsigned char c) {
                 return !isspace(c);
               }).base(),
               part.end());
    if (part.empty()) {
      *error = "invalid empty GPU id in \"" + raw + "\"";
      return std::nullopt;
    }
    char* parse_end = nullptr;
    const long parsed = strtol(part.c_str(), &parse_end, 10);
    if (parse_end == part.c_str() || *parse_end != '\0' || parsed < 0 ||
        parsed > INT32_MAX) {
      *error = "invalid GPU id \"" + part + "\"";
      return std::nullopt;
    }
    const int id = static_cast<int>(parsed);
    if (!seen.insert(id).second) {
      *error = "duplicate GPU id " + std::to_string(id);
      return std::nullopt;
    }
    ids.push_back(id);
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return ids;
}

std::string IDsEnv(const std::vector<int>& ids) {
  std::ostringstream out;
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i != 0) out << ',';
    out << ids[i];
  }
  return out.str();
}

std::vector<int> NormalizeIDs(const std::vector<int>& ids) {
  std::vector<int> out;
  std::set<int> seen;
  for (const int id : ids) {
    if (id < 0) continue;
    if (seen.insert(id).second) out.push_back(id);
  }
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<int> DiscoverGPUIDs() {
  auto from_smi = DiscoverGPUIDsWithNvidiaSMI();
  if (from_smi && !from_smi->empty()) {
    return *from_smi;
  }
  return DiscoverGPUIDsFromDev();
}

std::string EncodeRequest(const Request& request) {
  std::ostringstream out;
  out << "{\"action\":\"" << JsonEscape(request.action) << "\"";
  if (!request.ids.empty()) out << ",\"ids\":" << EncodeIDs(request.ids);
  if (request.count > 0) out << ",\"count\":" << request.count;
  if (request.wait) out << ",\"wait\":true";
  out << "}\n";
  return out.str();
}

std::string EncodeResponse(const Response& response) {
  std::ostringstream out;
  out << "{\"ok\":" << (response.ok ? "true" : "false");
  if (!response.error.empty()) {
    out << ",\"error\":\"" << JsonEscape(response.error) << "\"";
  }
  if (!response.lease.empty()) {
    out << ",\"lease\":\"" << JsonEscape(response.lease) << "\"";
  }
  if (!response.ids.empty()) {
    out << ",\"ids\":" << EncodeIDs(response.ids);
  }
  if (!response.leases.empty()) {
    out << ",\"leases\":{";
    bool first = true;
    for (const auto& [id, lease] : response.leases) {
      if (!first) out << ',';
      first = false;
      out << "\"" << JsonEscape(id) << "\":\"" << JsonEscape(lease) << "\"";
    }
    out << '}';
  }
  out << "}\n";
  return out.str();
}

std::optional<Request> DecodeRequest(const std::string& json,
                                     std::string* error) {
  JsonParser parser(json);
  auto value = parser.Parse(error);
  if (!value) return std::nullopt;
  if (value->type != JsonValue::Type::kObject) {
    *error = "request must be a JSON object";
    return std::nullopt;
  }

  Request request;
  for (const auto& [key, item] : value->object) {
    if (key == "action") {
      if (item.type != JsonValue::Type::kString) {
        *error = "action must be a string";
        return std::nullopt;
      }
      request.action = item.string;
    } else if (key == "ids") {
      auto ids = DecodeIDs(item, error);
      if (!ids) return std::nullopt;
      request.ids = std::move(*ids);
    } else if (key == "count") {
      if (item.type != JsonValue::Type::kNumber || item.number > INT32_MAX ||
          item.number < INT32_MIN) {
        *error = "count must be an integer";
        return std::nullopt;
      }
      request.count = static_cast<int>(item.number);
    } else if (key == "wait") {
      if (item.type != JsonValue::Type::kBool) {
        *error = "wait must be a boolean";
        return std::nullopt;
      }
      request.wait = item.boolean;
    }
  }
  return request;
}

std::optional<Response> DecodeResponse(const std::string& json,
                                       std::string* error) {
  JsonParser parser(json);
  auto value = parser.Parse(error);
  if (!value) return std::nullopt;
  if (value->type != JsonValue::Type::kObject) {
    *error = "response must be a JSON object";
    return std::nullopt;
  }

  Response response;
  for (const auto& [key, item] : value->object) {
    if (key == "ok") {
      if (item.type != JsonValue::Type::kBool) {
        *error = "ok must be a boolean";
        return std::nullopt;
      }
      response.ok = item.boolean;
    } else if (key == "error") {
      if (item.type != JsonValue::Type::kString) {
        *error = "error must be a string";
        return std::nullopt;
      }
      response.error = item.string;
    } else if (key == "lease") {
      if (item.type != JsonValue::Type::kString) {
        *error = "lease must be a string";
        return std::nullopt;
      }
      response.lease = item.string;
    } else if (key == "ids") {
      auto ids = DecodeIDs(item, error);
      if (!ids) return std::nullopt;
      response.ids = std::move(*ids);
    } else if (key == "leases") {
      if (item.type != JsonValue::Type::kObject) {
        *error = "leases must be an object";
        return std::nullopt;
      }
      for (const auto& [id, lease] : item.object) {
        if (lease.type != JsonValue::Type::kString) {
          *error = "lease values must be strings";
          return std::nullopt;
        }
        response.leases[id] = lease.string;
      }
    }
  }
  return response;
}

std::string ErrnoString(int err) {
  return std::string(strerror(err));
}

bool SendAll(int fd, const std::string& data, std::string* error) {
  size_t written = 0;
  while (written < data.size()) {
    const ssize_t n =
        write(fd, data.data() + written, data.size() - written);
    if (n > 0) {
      written += static_cast<size_t>(n);
      continue;
    }
    if (n < 0 && errno == EINTR) continue;
    *error = ErrnoString(errno);
    return false;
  }
  return true;
}

std::optional<std::string> ReadJsonLine(int fd, std::string* error) {
  std::string out;
  char c;
  while (true) {
    const ssize_t n = read(fd, &c, 1);
    if (n == 1) {
      if (c == '\n') return out;
      out.push_back(c);
      continue;
    }
    if (n == 0) {
      if (!out.empty()) return out;
      *error = "connection closed before response";
      return std::nullopt;
    }
    if (errno == EINTR) continue;
    *error = ErrnoString(errno);
    return std::nullopt;
  }
}

}  // namespace gpu_lease
