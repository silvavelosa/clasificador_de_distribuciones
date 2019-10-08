#ifndef CLASIFICADOR_DE_DISTRIBUCIONES_SECUENCIAL_IMPLEMENTACION_MANEJADOR_DE_ARCHIVOS_SECUENCIAL_H_
#define CLASIFICADOR_DE_DISTRIBUCIONES_SECUENCIAL_IMPLEMENTACION_MANEJADOR_DE_ARCHIVOS_SECUENCIAL_H_

#include <map>
#include <memory>
#include <vector>

#include "definiciones_compartidas/entidades.h"
#include "definiciones_compartidas/interfaces.h"

namespace clasificador_de_distribuciones
{
using namespace definiciones_compartidas;
namespace secuencial
{
namespace implementacion
{

class ManejadorDeArchivosSecuencial: public IManejadorDeArchivos
{
 public:
    int CargarDatos( const std::string& archivo,
            std::unique_ptr<std::vector<Validacion> >& validaciones,
            std::string& msg);

    int GenerarSalida( const std::string& archivo,
            const std::map<int,Distribucion>& ciudadano,
            std::string& msg);

    int GenerarSalida( const std::string& archivo,
            const std::vector<std::map<int,Distribucion>::iterator>& indice,
            std::string& msg);

 private:
    int TamanoDeArchivo (const std::string& archivo);
};

} // namespace implementacion
} // namespace secuencial
} // namespace clasificador_de_distribuciones

#endif // CLASIFICADOR_DE_DISTRIBUCIONES_SECUENCIAL_IMPLEMENTACION_MANEJADOR_DE_ARCHIVOS_SECUENCIAL_H_
