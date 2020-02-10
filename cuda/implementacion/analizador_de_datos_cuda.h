#ifndef CLASIFICADOR_DE_DISTRIBUCIONES_CUDA_IMPLEMENTACION_ANALIZADOR_DE_DATOS_CUDA_H_
#define CLASIFICADOR_DE_DISTRIBUCIONES_CUDA_IMPLEMENTACION_ANALIZADOR_DE_DATOS_CUDA_H_

#include <map>
#include <memory>
#include <vector>

#include "componentes_compartidos/entidades.h"
#include "componentes_compartidos/interfaces.h"
#include "secuencial/implementacion/analizador_de_datos_secuencial.h"

namespace clasificador_de_distribuciones
{
using namespace componentes_compartidos;
namespace cuda
{
namespace implementacion
{

class AnalizadorDeDatosCuda: public IAnalizadorDeDatos
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
            std::vector<Distribucion>& grupos) {
        return analizador_sec.RegresionLineal(grupos);
    }

    int OrdenarDistribuciones(
            std::unique_ptr<std::vector<Distribucion> >& grupos);
 private:
    secuencial::implementacion::AnalizadorDeDatosSecuencial analizador_sec;
};
} // namespace implementacion
} // namespace cuda
} // namespace clasificador_de_distribuciones

#endif // CLASIFICADOR_DE_DISTRIBUCIONES_CUDA_IMPLEMENTACION_ANALIZADOR_DE_DATOS_CUDA_H_
