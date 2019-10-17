#ifndef CLASIFICADOR_DE_DISTRIBUCIONES_OPEN_MP_IMPLEMENTACION_MANEJADOR_DE_ARCHIVOS_OPEN_MP_H_
#define CLASIFICADOR_DE_DISTRIBUCIONES_OPEN_MP_IMPLEMENTACION_MANEJADOR_DE_ARCHIVOS_OPEN_MP_H_

#include <map>
#include <memory>
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
};
} // namespace implementacion
} // namespace open_mp
} // namespace clasificador_de_distribuciones

#endif // CLASIFICADOR_DE_DISTRIBUCIONES_OPEN_MP_IMPLEMENTACION_MANEJADOR_DE_ARCHIVOS_OPEN_MP_H_
