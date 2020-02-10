#include "cuda/implementacion/kernel.h"

#include <fstream>
#include <iostream>
#include <omp.h>
#include <sstream>

#include "componentes_compartidos/util_archivos.h"
#include "cuda/implementacion/util_cuda.h"

#define CARACTERES_X_HILO (size_t)50
#define LINEAS_X_HILO_EST (size_t)13
#define HILOS_X_BLOQUE (size_t)256

namespace clasificador_de_distribuciones
{
namespace cuda
{
namespace implementacion
{

__device__ void InicializarVariablesCompartidas(size_t* n_eventos_bloque, 
                                            short* estado_bloque,
                                            size_t* posicion_error_bloque)
{
    *n_eventos_bloque = 0;
    *estado_bloque = 0;
    *posicion_error_bloque = 0;
}

__device__ void CopiarSeccionDeArchivo(size_t hilo, 
                                    size_t ini, size_t fin,
                                    size_t desplazamiento,
                                    char* archivo_gl, size_t l_archivo,
                                    char* texto_bloque)
{
    if(hilo == HILOS_X_BLOQUE - 1)
        fin += 25; 
    for(size_t i = ini; i < fin && i + desplazamiento < l_archivo ; i++)
    {
        texto_bloque[i] = archivo_gl[i+desplazamiento];
    }
}

__device__ size_t SiguienteSalto (size_t i, size_t l_bloque, char* texto_bloque)
{
    for(; i<=l_bloque && texto_bloque[i] != '\n' && texto_bloque[i] != '\r';
        ++i);
    if( texto_bloque[i] == '\r' && texto_bloque[i+1] == '\n' )
        ++i;
    return ++i;
}

__device__ void ProcesarSeccion ( size_t ini, size_t fin,
                                    char* texto_bloque, 
                                    Evento* eventos_bloque,
                                    size_t* n_eventos_bloque,
                                    short* estado_bloque)
{

    int id_grupo;
    int id_resultado;
    for ( size_t i = ini; i <= fin && *estado_bloque == 0 ;i++ )
    {
        id_grupo = id_resultado = 0;
        for(; texto_bloque[i] >= '0' && texto_bloque[i] <= '9'; i++)
        {
            id_grupo*=10;
            id_grupo+=texto_bloque[i]-'0';
        }
        i++;
        for(; texto_bloque[i] >= '0' && texto_bloque[i] <= '9'; i++)
        {        
            id_resultado*=10;
            id_resultado+=texto_bloque[i]-'0';
        }
        
        if(texto_bloque[i] == '\r' && texto_bloque[i+1] == '\n')
        {
            i++;
        }   
        unsigned int pos = atomicInc( (unsigned int*) n_eventos_bloque, 
                                HILOS_X_BLOQUE * LINEAS_X_HILO_EST );
        eventos_bloque[pos].id_grupo_ = id_grupo;
        eventos_bloque[pos].valor_ = id_resultado;
    }
}

__device__ void CopiarEventosBloque (size_t hilo, size_t bloque, 
                                        Evento* eventos_bloque,
                                        size_t n_eventos_bloque,
                                        Evento* eventos, size_t ancho_eventos,
                                        size_t* n_eventos)
{

    size_t eventos_x_hilo = n_eventos_bloque / HILOS_X_BLOQUE;

    if(hilo == 0)
    {
        n_eventos[bloque] = n_eventos_bloque;
    }
    
    Evento* fila_eventos = (Evento*)((char*)eventos + bloque * ancho_eventos);
    for( size_t i=eventos_x_hilo*hilo; i < eventos_x_hilo*(hilo+1) ; i++ )
        fila_eventos[i] = eventos_bloque[i];
    if(hilo == HILOS_X_BLOQUE - 1)
    {
        for(size_t i=eventos_x_hilo*(hilo+1); i < n_eventos_bloque; i++)
        {
            fila_eventos[i] = eventos_bloque[i];
        }
    }
}

__global__ void ProcesarArchivo (char* archivo, size_t l_archivo, 
                                    char separador,
                                    Evento* eventos, size_t ancho_eventos,
                                    size_t* n_eventos,
                                    short* estado, int* posicion_error)
{

    __shared__ size_t n_eventos_bloque;
    __shared__ short estado_bloque;
    __shared__ size_t posicion_error_bloque;
    
    __shared__ char texto_bloque[HILOS_X_BLOQUE * CARACTERES_X_HILO + 25];
    __shared__ Evento eventos_bloque[HILOS_X_BLOQUE * LINEAS_X_HILO_EST];

    size_t hilo = threadIdx.x;
    size_t bloque = blockIdx.x;

    size_t ini = CARACTERES_X_HILO * hilo;
    size_t fin = CARACTERES_X_HILO * (hilo + 1);
    size_t desplazamiento = CARACTERES_X_HILO * HILOS_X_BLOQUE * bloque;

    if(hilo == 0)
    { 
        InicializarVariablesCompartidas(&n_eventos_bloque,
                                        &estado_bloque,
                                        &posicion_error_bloque);
    }
    __syncthreads();

    CopiarSeccionDeArchivo(hilo, 
                        ini, fin, 
                        desplazamiento, 
                        archivo, l_archivo,
                        texto_bloque);

    __syncthreads(); 
    size_t l_bloque = min (CARACTERES_X_HILO*HILOS_X_BLOQUE + 25U, 
                            l_archivo - desplazamiento);
    fin = min(fin, l_bloque-1);
    if(ini < l_bloque)
    {
        if(hilo != 0 || bloque != 0)
        {
            ini = SiguienteSalto(ini, l_bloque, texto_bloque);
        }
        ProcesarSeccion (ini, fin,
                        texto_bloque,
                        eventos_bloque,
                        &n_eventos_bloque,
                        &estado_bloque);
    }    
    __syncthreads();  

    CopiarEventosBloque(hilo, bloque,
                        eventos_bloque,
                        n_eventos_bloque,
                        eventos, ancho_eventos,
                        n_eventos);
}

int ProcesarArchivoGPU (string archivo, char separador,
                unique_ptr<vector<Evento> >& eventos, string& msg)
{
    size_t tamano = TamanoDeArchivo(archivo);

    if( tamano == 0 || tamano == (size_t) -1 )
    {
        eventos.reset(nullptr);
        msg = "El archivo no pudo ser leído o está vacío";
        return -1;
    }

    std::ifstream entrada(archivo, std::ios::binary | std::ios::in);

    size_t n_bloques = tamano / (CARACTERES_X_HILO * HILOS_X_BLOQUE);
    if(n_bloques * CARACTERES_X_HILO * HILOS_X_BLOQUE < tamano)
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
        LINEAS_X_HILO_EST * HILOS_X_BLOQUE * sizeof(Evento),
        n_bloques);

    
    if(checkCUDAError("Reservando eventos GPU", msg))
        return -2;

    ProcesarArchivo <<<n_bloques, HILOS_X_BLOQUE >>> (archivo_device, tamano,
                                    separador,
                                    eventos_device, ancho_eventos_device,
                                    n_eventos_device,
                                    estado_device, posicion_error_device);

                                    
    eventos.reset(new vector<Evento>(
        n_bloques*LINEAS_X_HILO_EST * HILOS_X_BLOQUE));    
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
} // namespace cuda
} // namespace clasificador_de_distribuciones

using clasificador_de_distribuciones::componentes_compartidos::Evento;
using namespace clasificador_de_distribuciones::cuda::implementacion;
using std::cout;
using std::endl;
using std::string;
using std::unique_ptr;
using std::vector;

int main (int argc, char** argv)
{
    unique_ptr<vector<Evento> > eventos;
    string msg;
    if(ProcesarArchivoGPU(string(argv[1]), ';', eventos, msg) == 0)
    {
        cout<<eventos->size()<<endl;
    }
    return 0;
}