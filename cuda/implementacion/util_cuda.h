#ifndef CLASIFICADOR_DE_DISTRIBUCIONES_CUDA_IMPLEMENTACION_UTIL_CUDA_H_
#define CLASIFICADOR_DE_DISTRIBUCIONES_CUDA_IMPLEMENTACION_UTIL_CUDA_H_
#include <cuda_runtime.h>
#include <sstream>
#include <string>

namespace {
bool checkCUDAError(const char *fase, std::string& msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        std::stringstream ss;
        ss<<"Cuda error: "<<fase<<": "<<cudaGetErrorString(err)<<".\n";
        msg = ss.str();
        return true;
    }
    return false;
}
}

#endif //CLASIFICADOR_DE_DISTRIBUCIONES_CUDA_IMPLEMENTACION_UTIL_CUDA_H_