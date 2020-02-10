#include "cuda/implementacion/kernel.h"

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
    if(hilo == kHilosXBloque - 1)
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
                                kHilosXBloque * kLineasXHiloEst );
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

    size_t eventos_x_hilo = n_eventos_bloque / kHilosXBloque;

    if(hilo == 0)
    {
        n_eventos[bloque] = n_eventos_bloque;
    }
    
    Evento* fila_eventos = (Evento*)((char*)eventos + bloque * ancho_eventos);
    for( size_t i=eventos_x_hilo*hilo; i < eventos_x_hilo*(hilo+1) ; i++ )
        fila_eventos[i] = eventos_bloque[i];
    if(hilo == kHilosXBloque - 1)
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
    
    __shared__ char texto_bloque[kHilosXBloque * kCaracteresXHilo + 25];
    __shared__ Evento eventos_bloque[kHilosXBloque * kLineasXHiloEst];

    size_t hilo = threadIdx.x;
    size_t bloque = blockIdx.x;

    size_t ini = kCaracteresXHilo * hilo;
    size_t fin = kCaracteresXHilo * (hilo + 1);
    size_t desplazamiento = kCaracteresXHilo * kHilosXBloque * bloque;

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
    size_t l_bloque = min (kCaracteresXHilo*kHilosXBloque + 25U, 
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
} // namespace implementacion
} // namespace cuda
} // namespace clasificador_de_distribuciones