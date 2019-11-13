#ifndef CLASIFICADOR_DE_DISTRIBUCIONES_OPEN_MP_IMPLEMENTACION_MANEJADOR_DE_ARCHIVOS_OPEN_MP_H_
#define CLASIFICADOR_DE_DISTRIBUCIONES_OPEN_MP_IMPLEMENTACION_MANEJADOR_DE_ARCHIVOS_OPEN_MP_H_

#include <map>
#include <memory>
#include <omp.h>
#include <vector>

#include "componentes_compartidos/entidades.h"
#include "componentes_compartidos/interfaces.h"

namespace clasificador_de_distribuciones
{
using namespace componentes_compartidos;
namespace open_mp
{
namespace implementacion
{

class ManejadorDeArchivosOpenMP: public IManejadorDeArchivos
{
 public:
    ManejadorDeArchivosOpenMP(unsigned int n_hilos)
    {
     n_hilos_ = n_hilos;
    }

    ManejadorDeArchivosOpenMP(): ManejadorDeArchivosOpenMP(omp_get_num_procs()) {}
    int CargarDatos( const std::string& archivo,
            std::unique_ptr<std::vector<Evento> >& eventos,
            std::string& msg);

    int GenerarSalida( const std::string& archivo,
            const std::vector<Distribucion>& indice,
            std::string& msg,
            IManejadorDeArchivos::ModoDeEscritura modo
                = IManejadorDeArchivos::ModoDeEscritura::mantener);

 private:
    int TamanoDeArchivo (const std::string& archivo);
    unsigned int n_hilos_;
};
} // namespace implementacion
} // namespace open_mp
} // namespace clasificador_de_distribuciones

#endif // CLASIFICADOR_DE_DISTRIBUCIONES_OPEN_MP_IMPLEMENTACION_MANEJADOR_DE_ARCHIVOS_OPEN_MP_H_
