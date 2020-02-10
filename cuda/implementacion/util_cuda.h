#ifndef CLASIFICADOR_DE_DISTRIBUCIONES_CUDA_IMPLEMENTACION_UTIL_CUDA_H_
#define CLASIFICADOR_DE_DISTRIBUCIONES_CUDA_IMPLEMENTACION_UTIL_CUDA_H_

#include <string>

namespace clasificador_de_distribuciones
{
namespace cuda
{
namespace implementacion
{
bool checkCUDAError(const char *fase, std::string& msg);
} // namespace implementacion
} // namespace secuencial
} // namespace clasificador_de_distribuciones
#endif //CLASIFICADOR_DE_DISTRIBUCIONES_CUDA_IMPLEMENTACION_UTIL_CUDA_H_