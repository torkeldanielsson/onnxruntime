#pragma once
#include "core/common/common.h"
#include "core/session/onnxruntime_c_api.h"

ONNXTensorElementDataType MLDataTypeToOnnxRuntimeTensorElementDataType(const onnxruntime::DataTypeImpl* cpp_type);
namespace onnxruntime {

using DType = ONNXTensorElementDataType;

typedef struct {
  void* data;
  /*! \brief Number of dimensions */
  size_t ndim;
  /*! \brief The data type of the pointer*/
  DType dtype;
  /*! \brief The shape of the tensor */
  int64_t* shape;
} ONNXRunTimeTensor;

// AllocateFunc(void* handle, size_t alignment, size_t size)
using AllocateFunc = void* (*)(void*, size_t, size_t);
using DestroyFunc = void (*)(void*, void*);
using AllocatorHandle = void*;

typedef struct {
  //right now we only include allocation for host memory
  AllocateFunc allocate_func;
  DestroyFunc release_func;
  AllocatorHandle allocator_handle;
  const char* node_name;
} ComputeContext;

using FunctionState = void*;
// take the ComputeContext, and create a function state.
using CreateFunctionStateC = int (*)(ComputeContext*, FunctionState*);
// pass in the function state and input/output tensors, perform compute and return status code, 0 - succeed.
using ComputeFuncC = int (*)(FunctionState, ONNXRunTimeTensor*, size_t, ONNXRunTimeTensor*, size_t);
// release the function state.
using DestroyFunctionStateC = void (*)(FunctionState);
}  // namespace onnxruntime
