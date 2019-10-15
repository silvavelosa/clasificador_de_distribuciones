#ifndef CLASIFICADOR_DE_DISTRIBUCIONES_COMPONENTES_COMPARTIDOS_INTERFACES_H_
#define CLASIFICADOR_DE_DISTRIBUCIONES_COMPONENTES_COMPARTIDOS_INTERFACES_H_

#include <map>
#include <memory>
#include <vector>

#include "componentes_compartidos/entidades.h"

namespace clasificador_de_distribuciones
{
namespace componentes_compartidos
{
class IManejadorDeArchivos
{
 public:
    enum ModoDeEscritura{
     reemplazar,
     concatenar,
     mantener
    };
    virtual int CargarDatos( const std::string& archivo,
            std::unique_ptr<std::vector<Validacion> >& validaciones,
            std::string& msg) = 0;

    virtual int GenerarSalida( const std::string& archivo,
            const std::map<int,Distribucion>& ciudadanos,
            std::string& msg,
            ModoDeEscritura modo = ModoDeEscritura::mantener) = 0;

    virtual int GenerarSalida( const std::string& archivo,
            const std::vector<std::map<int,Distribucion>::iterator>& indice,
            std::string& msg,
            ModoDeEscritura modo = ModoDeEscritura::mantener) = 0;
};


class IAnalizadorDeDatos
{
 public:
    virtual int OrdenarValidaciones(
            std::unique_ptr<std::vector<Validacion> >& validaciones) = 0;

    virtual int AgruparYPromediar(
            const std::vector<Validacion>& valideaciones,
            std::unique_ptr<std::map<int,Distribucion> >& ciudadanos,
            std::unique_ptr<Distribucion>& promedio) = 0;

    virtual int CompararDistribuciones(
            const std::unique_ptr<std::map<int,Distribucion> >& ciudadanos,
            const Distribucion& promedio) = 0;

    virtual int OrdenarDistribuciones(
            const std::map<int,Distribucion>& ciudadanos,
            std::unique_ptr<
                std::vector<std::map<int,Distribucion>::iterator> >& indice) = 0;
};
} // namespace componentes_compartidos
} // namespace clasificador_de_versiones
#endif // CLASIFICADOR_DE_DISTRIBUCIONES_COMPONENTES_COMPARTIDOS_INTERFACES_H_
