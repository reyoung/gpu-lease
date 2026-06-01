#include "gpu_lease/app.h"

#include "gpu_lease/client.h"
#include "gpu_lease/common.h"
#include "gpu_lease/server.h"

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <utility>

namespace gpu_lease {
namespace {

void Usage(std::ostream& out) {
  out << "Usage:\n"
      << "  gpu-lease daemon [--socket PATH] [--busy-matmul] [PATH]\n"
      << "  gpu-lease run --ids 0,1 -- command [args...]\n"
      << "  gpu-lease run --count 2 [--wait] -- command [args...]\n"
      << "  gpu-lease status [--socket PATH]\n\n"
      << "Environment:\n"
      << "  GPU_LEASE_SOCKET overrides the default socket path ("
      << kDefaultSocketPath << ").\n";
}

bool ConsumeValue(const std::vector<std::string>& args, size_t* index,
                  const std::string& name, std::string* value,
                  std::ostream& err) {
  const std::string& arg = args[*index];
  const std::string prefix = name + "=";
  if (arg.rfind(prefix, 0) == 0) {
    *value = arg.substr(prefix.size());
    return true;
  }
  if (*index + 1 >= args.size()) {
    err << name << " requires a value\n";
    return false;
  }
  ++*index;
  *value = args[*index];
  return true;
}

bool ParseIntFlag(const std::string& raw, int* value) {
  char* end = nullptr;
  const long parsed = strtol(raw.c_str(), &end, 10);
  if (end == raw.c_str() || *end != '\0' || parsed < 0 ||
      parsed > INT32_MAX) {
    return false;
  }
  *value = static_cast<int>(parsed);
  return true;
}

int RunProcess(const std::vector<std::string>& command,
               const std::vector<int>& ids, std::ostream& err) {
  const pid_t pid = fork();
  if (pid < 0) {
    err << "run: command failed: " << ErrnoString(errno) << "\n";
    return 1;
  }
  if (pid == 0) {
    setenv("CUDA_VISIBLE_DEVICES", IDsEnv(ids).c_str(), 1);
    std::vector<char*> argv;
    argv.reserve(command.size() + 1);
    for (const std::string& item : command) {
      argv.push_back(const_cast<char*>(item.c_str()));
    }
    argv.push_back(nullptr);
    execvp(argv[0], argv.data());
    std::cerr << "run: command failed: " << ErrnoString(errno) << "\n";
    _exit(127);
  }

  int status = 0;
  while (waitpid(pid, &status, 0) < 0) {
    if (errno == EINTR) continue;
    err << "run: command failed: " << ErrnoString(errno) << "\n";
    return 1;
  }
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  if (WIFSIGNALED(status)) {
    return 128 + WTERMSIG(status);
  }
  return 1;
}

int DaemonCommand(const std::vector<std::string>& args, std::ostream& out,
                  std::ostream& err) {
  std::string socket_flag;
  std::string positional;
  bool busy_matmul = false;
  bool have_positional = false;

  for (size_t i = 0; i < args.size(); ++i) {
    const std::string& arg = args[i];
    if (arg == "--") {
      err << "daemon accepts at most one socket path\n";
      return 2;
    }
    if (arg == "--socket" || arg.rfind("--socket=", 0) == 0) {
      if (!ConsumeValue(args, &i, "--socket", &socket_flag, err)) return 2;
    } else if (arg == "--busy-matmul") {
      busy_matmul = true;
    } else if (arg == "-h" || arg == "--help") {
      Usage(out);
      return 0;
    } else if (!arg.empty() && arg[0] == '-') {
      err << "unknown daemon flag " << arg << "\n";
      return 2;
    } else {
      if (have_positional) {
        err << "daemon accepts at most one socket path\n";
        return 2;
      }
      positional = arg;
      have_positional = true;
    }
  }

  std::string socket_path = SocketPath(socket_flag);
  if (have_positional) {
    socket_path = positional;
  }

  ServerOptions options;
  options.available_ids = DiscoverGPUIDs();
  options.busy_matmul = busy_matmul;
  Server server(std::move(options));
  std::string error;
  if (!server.Run(socket_path, &error, true)) {
    err << "daemon: " << error << "\n";
    return 1;
  }
  return 0;
}

int RunCommand(const std::vector<std::string>& args, std::ostream& out,
               std::ostream& err) {
  std::string ids_raw;
  std::string socket_flag;
  int num = 0;
  int count = 0;
  bool wait = false;
  std::vector<std::string> command;

  for (size_t i = 0; i < args.size(); ++i) {
    const std::string& arg = args[i];
    if (arg == "--") {
      command.assign(args.begin() + static_cast<long>(i) + 1, args.end());
      break;
    }
    if (arg == "--ids" || arg.rfind("--ids=", 0) == 0) {
      if (!ConsumeValue(args, &i, "--ids", &ids_raw, err)) return 2;
    } else if (arg == "--socket" || arg.rfind("--socket=", 0) == 0) {
      if (!ConsumeValue(args, &i, "--socket", &socket_flag, err)) return 2;
    } else if (arg == "--num" || arg.rfind("--num=", 0) == 0) {
      std::string raw;
      if (!ConsumeValue(args, &i, "--num", &raw, err)) return 2;
      if (!ParseIntFlag(raw, &num)) {
        err << "run: invalid --num value " << raw << "\n";
        return 2;
      }
    } else if (arg == "--count" || arg.rfind("--count=", 0) == 0) {
      std::string raw;
      if (!ConsumeValue(args, &i, "--count", &raw, err)) return 2;
      if (!ParseIntFlag(raw, &count)) {
        err << "run: invalid --count value " << raw << "\n";
        return 2;
      }
    } else if (arg == "--wait") {
      wait = true;
    } else if (arg == "-h" || arg == "--help") {
      Usage(out);
      return 0;
    } else if (!arg.empty() && arg[0] == '-') {
      err << "unknown run flag " << arg << "\n";
      return 2;
    } else {
      command.assign(args.begin() + static_cast<long>(i), args.end());
      break;
    }
  }

  if (command.empty()) {
    err << "run requires a command after --\n";
    return 2;
  }
  if (num > 0 && count > 0 && num != count) {
    err << "run: --num and --count cannot specify different values\n";
    return 2;
  }
  int requested_count = num > 0 ? num : count;

  AcquireOptions options;
  options.count = requested_count;
  options.wait = wait;
  if (!ids_raw.empty()) {
    if (requested_count > 0) {
      err << "run: --ids cannot be used with --num or --count\n";
      return 2;
    }
    std::string parse_error;
    auto ids = ParseIDs(ids_raw, &parse_error);
    if (!ids) {
      err << "run: " << parse_error << "\n";
      return 2;
    }
    options.ids = std::move(*ids);
  } else if (requested_count <= 0) {
    err << "run: either --ids, --num, or --count must be specified\n";
    return 2;
  }

  AcquireResult acquired =
      AcquireWithOptions(SocketPath(socket_flag), options);
  if (!acquired.lease) {
    err << "run: acquire lease: " << acquired.error << "\n";
    return 1;
  }

  std::vector<int> held_ids = acquired.lease->ids();
  if (held_ids.empty()) {
    held_ids = options.ids;
  }
  const int code = RunProcess(command, held_ids, err);
  acquired.lease->Close();
  return code;
}

int StatusCommand(const std::vector<std::string>& args, std::ostream& out,
                  std::ostream& err) {
  std::string socket_flag;
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string& arg = args[i];
    if (arg == "--socket" || arg.rfind("--socket=", 0) == 0) {
      if (!ConsumeValue(args, &i, "--socket", &socket_flag, err)) return 2;
    } else if (arg == "-h" || arg == "--help") {
      Usage(out);
      return 0;
    } else {
      err << "unknown status flag " << arg << "\n";
      return 2;
    }
  }

  StatusResult status = GetStatus(SocketPath(socket_flag));
  if (!status.error.empty()) {
    err << "status: " << status.error << "\n";
    return 1;
  }

  std::vector<std::pair<int, std::string>> numeric;
  numeric.reserve(status.leases.size());
  for (const auto& [id, lease] : status.leases) {
    char* end = nullptr;
    const long parsed = strtol(id.c_str(), &end, 10);
    if (end != id.c_str() && *end == '\0') {
      numeric.push_back({static_cast<int>(parsed), lease});
    }
  }
  std::sort(numeric.begin(), numeric.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  for (const auto& [id, lease] : numeric) {
    out << id << " " << lease << "\n";
  }
  return 0;
}

}  // namespace

int RunCLI(const std::vector<std::string>& args, std::ostream& out,
           std::ostream& err) {
  if (args.empty()) {
    Usage(err);
    return 2;
  }

  const std::string& command = args[0];
  std::vector<std::string> rest(args.begin() + 1, args.end());
  if (command == "daemon" || command == "deamon") {
    return DaemonCommand(rest, out, err);
  }
  if (command == "run") {
    return RunCommand(rest, out, err);
  }
  if (command == "status") {
    return StatusCommand(rest, out, err);
  }
  if (command == "-h" || command == "--help" || command == "help") {
    Usage(out);
    return 0;
  }

  err << "unknown command \"" << command << "\"\n";
  Usage(err);
  return 2;
}

}  // namespace gpu_lease
