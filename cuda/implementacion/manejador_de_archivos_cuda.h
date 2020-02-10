#ifndef CLASIFICADOR_DE_DISTRIBUCIONES_CUDA_IMPLEMENTACION_MANEJADOR_DE_ARCHIVOS_CUDA_H_
#define CLASIFICADOR_DE_DISTRIBUCIONES_CUDA_IMPLEMENTACION_MANEJADOR_DE_ARCHIVOS_CUDA_H_

#include <map>
#include <memory>
#include <vector>

#include "componentes_compartidos/entidades.h"
#include "componentes_compartidos/interfaces.h"
#include "secuencial/implementacion/manejador_de_archivos_secuencial.h"

namespace clasificador_de_distribuciones
{
using namespace componentes_compartidos;
namespace cuda
{
namespace implementacion
{

class ManejadorDeArchivosCuda: public IManejadorDeArchivos
{
 public:
    int CargarDatos( const std::string& archivo,
            std::unique_ptr<std::vector<Evento> >& eventos,
            std::string& msg);

    int GenerarSalida( const std::string& archivo,
            const std::vector<Distribucion>& indice,
            std::string& msg,
            IManejadorDeArchivos::ModoDeEscritura modo
                = IManejadorDeArchivos::ModoDeEscritura::mantener);

 private:
    secuencial::implementacion::ManejadorDeArchivosSecuencial manejador_sec_;
};
} // namespace implementacion
} // namespace cuda
} // namespace clasificador_de_distribuciones

#endif // CLASIFICADOR_DE_DISTRIBUCIONES_CUDA_IMPLEMENTACION_MANEJADOR_DE_ARCHIVOS_CUDA_H_
