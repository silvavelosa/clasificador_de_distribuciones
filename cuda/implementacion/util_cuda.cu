#include "cuda/implementacion/util_cuda.h"

#include <cuda_runtime.h>
#include <sstream>

namespace clasificador_de_distribuciones
{
namespace cuda
{
namespace implementacion
{
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
} // namespace implementacion
} // namespace secuencial
} // namespace clasificador_de_distribuciones