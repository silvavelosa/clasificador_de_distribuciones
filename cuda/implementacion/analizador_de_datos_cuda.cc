#include "cuda/implementacion/analizador_de_datos_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <gsl/gsl_multifit.h>

namespace clasificador_de_distribuciones
{
namespace cuda
{
namespace implementacion
{
using std::map;
using std::sort;
using std::string;
using std::unique_ptr;
using std::vector;

int AnalizadorDeDatosCuda::OrdenarEventos(
        unique_ptr<vector<Evento> >& eventos) {
    sort(eventos->begin(), eventos->end());
    return 0;
}

int AnalizadorDeDatosCuda::AgruparYPromediar(
        const vector<Evento>& eventos,
        unique_ptr<vector<Distribucion> >& grupos,
        unique_ptr<Distribucion>& promedio) {
    promedio.reset(new Distribucion());
    grupos.reset(new vector<Distribucion> ());
    for(size_t i=0;i<eventos.size();i++)
    {
        if(grupos->empty() || grupos->back().Grupo() != eventos[i].id_grupo_)
        {
             grupos->push_back(Distribucion(eventos[i].id_grupo_));
        }
        grupos->back().AnadirEvento(eventos[i]);
        promedio->AnadirEvento(eventos[i]);
    }
    return 0;
}

int AnalizadorDeDatosCuda::CompararDistribuciones(
        vector<Distribucion>& grupos,
        const Distribucion& promedio) {
    for(size_t i=0;i<grupos.size();i++)
    {
        grupos[i].EstablecerDiferencia(promedio);
    }
    return 0;
}

bool OrdenarPorResiduo (const Distribucion& a,
                        const Distribucion& b) {
    return (a.Residuo() > b.Residuo());
}

int AnalizadorDeDatosCuda::OrdenarDistribuciones(
        unique_ptr<vector<Distribucion> >& grupos) {
        sort(grupos->begin(),grupos->end(),OrdenarPorResiduo);
    return 0;
}
} // namespace implementacion
} // namespace cuda
} // namespace clasificador_de_distribuciones

