#ifndef CLASIFICADOR_DE_DISTRIBUCIONES_SECUENCIAL_IMPLEMENTACION_MANEJADOR_DE_ARCHIVOS_SECUENCIAL_H_
#define CLASIFICADOR_DE_DISTRIBUCIONES_SECUENCIAL_IMPLEMENTACION_MANEJADOR_DE_ARCHIVOS_SECUENCIAL_H_

#include <memory>
#include <vector>

#include "componentes_compartidos/entidades.h"
#include "componentes_compartidos/interfaces.h"

namespace clasificador_de_distribuciones
{
using namespace componentes_compartidos;
namespace secuencial
{
namespace implementacion
{

class ManejadorDeArchivosSecuencial: public IManejadorDeArchivos
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
};
} // namespace implementacion
} // namespace secuencial
} // namespace clasificador_de_distribuciones

#endif // CLASIFICADOR_DE_DISTRIBUCIONES_SECUENCIAL_IMPLEMENTACION_MANEJADOR_DE_ARCHIVOS_SECUENCIAL_H_
