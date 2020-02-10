#include "cuda/implementacion/manejador_de_archivos_cuda.h"

#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <sstream>

#include "componentes_compartidos/util_archivos.h"
#include "cuda/implementacion/util_cuda.h"
#include "cuda/implementacion/kernel.h"

namespace clasificador_de_distribuciones
{
namespace cuda
{
namespace implementacion
{
using std::ios;
using std::map;
using std::max;
using std::min;
using std::string;
using std::unique_ptr;
using std::vector;

int ManejadorDeArchivosCuda::CargarDatos(const string& archivo,
        unique_ptr<vector<Evento> >& eventos,
        string& msg)
{
    size_t tamano = TamanoDeArchivo(archivo);

    if( tamano == 0 || tamano == (size_t) -1 )
    {
        eventos.reset(nullptr);
        msg = "El archivo no pudo ser leído o está vacío";
        return -1;
    }

    if(tamano < 10000000)
    {
        return manejador_sec_.CargarDatos(archivo, eventos, msg);
    }

    std::ifstream entrada(archivo, std::ios::binary | std::ios::in);

    size_t n_bloques = tamano / (kCaracteresXHilo * kHilosXBloque);
    if(n_bloques * kCaracteresXHilo * kHilosXBloque < tamano)
        n_bloques++;

    char* archivo_device;
    cudaMalloc( (void**) &archivo_device, tamano*sizeof(char));
    char* archivo_host = (char*) malloc(tamano*sizeof(char));
    std::cout<<"Leyendo Archivo"<<std::endl;

    entrada.read(archivo_host, tamano);

    std::cout<<"Copiando Archivo"<<std::endl;
    if(checkCUDAError("Reservando archivo GPU", msg))
        return -2;

    cudaMemcpy (archivo_device, archivo_host, tamano*sizeof(char),
                    cudaMemcpyHostToDevice);
    
    
    if(checkCUDAError("Copiando archivo a GPU", msg))
        return -2;


    size_t ancho_eventos_device;
    Evento* eventos_device;
    size_t* n_eventos_device;
    short* estado_device;
    int* posicion_error_device;
    
    cudaMalloc (&estado_device, sizeof(short));
    cudaMalloc (&posicion_error_device, sizeof(int));
    cudaMalloc (&n_eventos_device, n_bloques * sizeof(size_t));
    cudaMallocPitch(&eventos_device, &ancho_eventos_device,
        kLineasXHiloEst * kHilosXBloque * sizeof(Evento),
        n_bloques);

    
    if(checkCUDAError("Reservando eventos GPU", msg))
        return -2;

    ProcesarArchivo <<<n_bloques, kHilosXBloque >>> (archivo_device, tamano,
                                    ';',
                                    eventos_device, ancho_eventos_device,
                                    n_eventos_device,
                                    estado_device, posicion_error_device);

                                    
    eventos.reset(new vector<Evento>(
        n_bloques*kLineasXHiloEst * kHilosXBloque));    
    size_t* n_eventos_host = (size_t*) malloc(sizeof(size_t)*n_bloques);

    if(checkCUDAError("Procesando Archivo",msg))
        return -2;

    cudaMemcpy(n_eventos_host, n_eventos_device, sizeof(size_t)*n_bloques,
                cudaMemcpyDeviceToHost);
    
    if(checkCUDAError("Copiando n_eventos a CPU",msg))
        return -2;
    
    size_t total_eventos=0;
    for(size_t i=0;i<n_bloques;i++)
    {
        cudaMemcpy(eventos->data() + total_eventos, 
            (char*) eventos_device + (i*ancho_eventos_device),
            sizeof(Evento)*n_eventos_host[i],
            cudaMemcpyDeviceToHost);
        total_eventos+= n_eventos_host[i];
    }
    eventos->resize(total_eventos);
    return 0;
}
} // namespace implementacion
} // namespace open_mp
} // namespace clasificador_de_distribuciones
