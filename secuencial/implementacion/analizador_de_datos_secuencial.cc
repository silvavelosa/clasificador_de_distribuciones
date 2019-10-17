#include "secuencial/implementacion/analizador_de_datos_secuencial.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <gsl/gsl_multifit.h>

namespace clasificador_de_distribuciones
{
namespace secuencial
{
namespace implementacion
{
using std::map;
using std::sort;
using std::string;
using std::unique_ptr;
using std::vector;

int AnalizadorDeDatosSecuencial::OrdenarEventos(
        unique_ptr<vector<Evento> >& eventos) {
    sort(eventos->begin(), eventos->end());
    return 0;
}

int AnalizadorDeDatosSecuencial::AgruparYPromediar(
        const vector<Evento>& eventos,
        unique_ptr<vector<Distribucion> >& grupos,
        unique_ptr<Distribucion>& promedio) {
    promedio.reset(new Distribucion());
    grupos.reset(new vector<Distribucion> ());
    for(unsigned int i=0;i<eventos.size();i++)
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

int AnalizadorDeDatosSecuencial::CompararDistribuciones(
        vector<Distribucion>& grupos,
        const Distribucion& promedio) {
    for(unsigned int i=0;i<grupos.size();i++)
    {
        grupos[i].EstablecerDiferencia(promedio);
    }
    return 0;
}


int AnalizadorDeDatosSecuencial::RegresionLineal(
        vector<Distribucion>& grupos) {
    return -3;
}

bool OrdenarPorResiduo (const Distribucion& a,
                        const Distribucion& b) {
    return (a.Residuo() < b.Residuo());
}

int AnalizadorDeDatosSecuencial::OrdenarDistribuciones(
        unique_ptr<vector<Distribucion> >& grupos) {
        sort(grupos->begin(),grupos->end(),OrdenarPorResiduo);
    return 0;
}
} // namespace implementacion
} // namespace secuencial
} // namespace clasificador_de_distribuciones

