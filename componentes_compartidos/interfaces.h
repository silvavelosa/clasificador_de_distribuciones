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
            std::unique_ptr<std::vector<Evento> >& eventos,
            std::string& msg) = 0;

    virtual int GenerarSalida( const std::string& archivo,
            const std::map<int,Distribucion>& ciudadanos,
            std::string& msg,
            ModoDeEscritura modo = ModoDeEscritura::mantener) = 0;

    virtual int GenerarSalida( const std::string& archivo,
            const std::vector<std::map<int,Distribucion>::const_iterator>& indice,
            std::string& msg,
            ModoDeEscritura modo = ModoDeEscritura::mantener) = 0;
};


class IAnalizadorDeDatos
{
 public:
    virtual int OrdenarEventos(
            std::unique_ptr<std::vector<Evento> >& eventos) = 0;

    virtual int AgruparYPromediar(
            const std::vector<Evento>& eventos,
            std::unique_ptr<std::map<int,Distribucion> >& ciudadanos,
            std::unique_ptr<Distribucion>& promedio) = 0;

    virtual int CompararDistribuciones(
            const std::unique_ptr<std::map<int,Distribucion> >& ciudadanos,
            const Distribucion& promedio) = 0;

    virtual int RegresionLineal(
            const std::unique_ptr<std::map<int,Distribucion> >& ciudadanos);

    virtual int OrdenarDistribuciones(
            const std::map<int,Distribucion>& ciudadanos,
            std::unique_ptr<
                std::vector<std::map<int,Distribucion>::const_iterator> >& indice) = 0;
};
} // namespace componentes_compartidos
} // namespace clasificador_de_versiones
#endif // CLASIFICADOR_DE_DISTRIBUCIONES_COMPONENTES_COMPARTIDOS_INTERFACES_H_
