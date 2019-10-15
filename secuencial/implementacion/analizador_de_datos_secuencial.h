#ifndef CLASIFICADOR_DE_DISTRIBUCIONES_SECUENCIAL_IMPLEMENTACION_ANALIZADOR_DE_DATOS_SECUENCIAL_H_
#define CLASIFICADOR_DE_DISTRIBUCIONES_SECUENCIAL_IMPLEMENTACION_ANALIZADOR_DE_DATOS_SECUENCIAL_H_

#include <map>
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

class AnalizadorDeDatosSecuencial: public IAnalizadorDeDatos
{
 public:
    int OrdenarValidaciones(
            std::unique_ptr<std::vector<Validacion> >& validaciones);

    int AgruparYPromediar(
            const std::vector<Validacion>& valideaciones,
            std::unique_ptr<std::map<int,Distribucion> >& ciudadanos,
            Distribucion& promedio);

    int CompararDistribuciones(
            const std::unique_ptr<std::map<int,Distribucion> >& ciudadanos,
            const Distribucion& promedio);

    int OrdenarDistribuciones(
            const std::map<int,Distribucion>& ciudadanos,
            std::unique_ptr<std::vector<std::map<int,Distribucion>::iterator> >& indice);
};
} // namespace implementacion
} // namespace secuencial
} // namespace clasificador_de_distribuciones

#endif // CLASIFICADOR_DE_DISTRIBUCIONES_SECUENCIAL_IMPLEMENTACION_ANALIZADOR_DE_DATOS_SECUENCIAL_H_
