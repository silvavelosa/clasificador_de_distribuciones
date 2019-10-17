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
    int OrdenarEventos(
            std::unique_ptr<std::vector<Evento> >& eventos);

    int AgruparYPromediar(
            const std::vector<Evento>& eventos,
            std::unique_ptr<std::vector<Distribucion> >& grupos,
            std::unique_ptr<Distribucion>& promedio);

    int CompararDistribuciones(
            std::vector<Distribucion>& grupos,
            const Distribucion& promedio);

    int RegresionLineal(
            std::vector<Distribucion>& grupos);

    int OrdenarDistribuciones(
            std::unique_ptr<std::vector<Distribucion> >& grupos);
};
} // namespace implementacion
} // namespace secuencial
} // namespace clasificador_de_distribuciones

#endif // CLASIFICADOR_DE_DISTRIBUCIONES_SECUENCIAL_IMPLEMENTACION_ANALIZADOR_DE_DATOS_SECUENCIAL_H_
