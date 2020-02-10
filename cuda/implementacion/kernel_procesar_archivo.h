#ifndef CLASIFICADOR_DE_DISTRIBUCIONES_CUDA_IMPLEMENTACION_KERNEL_H_
#define CLASIFICADOR_DE_DISTRIBUCIONES_CUDA_IMPLEMENTACION_KERNEL_H_

#include "componentes_compartidos/entidades.h"

namespace clasificador_de_distribuciones
{
using namespace componentes_compartidos;
namespace cuda
{
namespace implementacion
{
    
const size_t kCaracteresXHilo = 50; 
const size_t kLineasXHiloEst = 13;
const size_t kHilosXBloque = 256;

__global__ void ProcesarArchivo (char* archivo, size_t l_archivo, 
                                    char separador,
                                    Evento* eventos, size_t ancho_eventos,
                                    size_t* n_eventos,
                                    short* estado, int* posicion_error);
} // namespace implementacion
} // namespace cuda
} // namespace clasificador_de_distribuciones
#endif // CLASIFICADOR_DE_DISTRIBUCIONES_CUDA_IMPLEMENTACION_KERNEL_H_