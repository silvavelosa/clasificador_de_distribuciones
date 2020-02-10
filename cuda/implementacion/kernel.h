#ifndef CLASIFICADOR_DE_DISTRIBUCIONES_CUDA_IMPLEMENTACION_KERNEL_H_
#define CLASIFICADOR_DE_DISTRIBUCIONES_CUDA_IMPLEMENTACION_KERNEL_H_

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

#include "componentes_compartidos/entidades.h"

namespace clasificador_de_distribuciones
{
using std::string;
using std::unique_ptr;
using std::vector;
using namespace componentes_compartidos;
namespace cuda
{
namespace implementacion
{
int ProcesarArchivoGPU (string archivo, char separador,
                unique_ptr<vector<Evento> >& eventos, string& msg);
} // namespace implementacion
} // namespace cuda
} // namespace clasificador_de_distribuciones
#endif // CLASIFICADOR_DE_DISTRIBUCIONES_CUDA_IMPLEMENTACION_KERNEL_H_