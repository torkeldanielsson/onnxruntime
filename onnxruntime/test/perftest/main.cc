// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// onnxruntime dependencies
#include <core/session/onnxruntime_c_api.h>
#include <random>
#include "command_args_parser.h"
#include "performance_runner.h"

using namespace onnxruntime;

#ifdef _WIN32
int real_main(int argc, wchar_t* argv[]) {
#else
int real_main(int argc, char* argv[]) {
#endif
  perftest::PerformanceTestConfig test_config;
  if (!perftest::CommandLineParser::ParseArguments(test_config, argc, argv)) {
    perftest::CommandLineParser::ShowUsage();
    return -1;
  }
  Ort::Env env{nullptr};
  try {
    OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_WARNING;
    env = Ort::Env(logging_level, "Default");
  } catch (const Ort::Exception& e) {
    fprintf(stderr, "Error creating environment: %s \n", e.what());
    return -1;
  }
  std::random_device rd;
  perftest::PerformanceRunner perf_runner(env, test_config, rd);
  auto status = perf_runner.Run();
  if (!status.IsOK()) {
    printf("Run failed:%s\n", status.ErrorMessage().c_str());
    return -1;
  }

  perf_runner.SerializeResult();

  return 0;
}

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
  int retval = -1;
  try {
    retval = real_main(argc, argv);
  } catch (std::exception& ex) {
    fprintf(stderr, "%s\n", ex.what());
    retval = -1;
  }
  return retval;
}
