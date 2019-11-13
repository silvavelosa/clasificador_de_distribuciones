#ifndef CLASIFICADOR_DE_DISTRIBUCIONES_OPEN_MP_IMPLEMENTACION_ANALIZADOR_DE_DATOS_OPEN_MP_H_
#define CLASIFICADOR_DE_DISTRIBUCIONES_OPEN_MP_IMPLEMENTACION_ANALIZADOR_DE_DATOS_OPEN_MP_H_

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

class AnalizadorDeDatosOpenMP: public IAnalizadorDeDatos
{
 public:
    AnalizadorDeDatosOpenMP(unsigned int n_hilos)
    {
        n_hilos_ = n_hilos;
    }
    AnalizadorDeDatosOpenMP(): AnalizadorDeDatosOpenMP(omp_get_num_procs()) {}
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
 private:
    unsigned int n_hilos_;
};
} // namespace implementacion
} // namespace open_mp
} // namespace clasificador_de_distribuciones

#endif // CLASIFICADOR_DE_DISTRIBUCIONES_OPEN_MP_IMPLEMENTACION_ANALIZADOR_DE_DATOS_OPEN_MP_H_
