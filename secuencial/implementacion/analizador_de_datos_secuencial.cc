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
        unique_ptr<map<int,Distribucion> >& ciudadanos,
        unique_ptr<Distribucion>& promedio) {
    promedio.reset(new Distribucion());
    ciudadanos.reset(new map<int,Distribucion> ());
    map<int,Distribucion>::iterator actual = ciudadanos->end();
    for(unsigned int i=0;i<eventos.size();i++)
    {
        if(actual == ciudadanos->end()
                || actual->first != eventos[i].id_grupo_)
        {
            actual = ciudadanos->emplace_hint(ciudadanos->end(),
                                      eventos[i].id_grupo_,Distribucion());
        }
        if(Distribucion::tamano_frecuencias_ <= eventos[i].valor_/10)
        {
            actual->second.frecuencias_[Distribucion::tamano_frecuencias_-1]++;
            promedio->frecuencias_[Distribucion::tamano_frecuencias_-1]++;
        }
        else
        {
            actual->second.frecuencias_[eventos[i].valor_/10]++;
            promedio->frecuencias_[eventos[i].valor_/10]++;
        }
        actual->second.total_++;
        promedio->total_++;
    }
    return 0;
}

int AnalizadorDeDatosSecuencial::CompararDistribuciones(
        const unique_ptr<map<int,Distribucion> >& ciudadanos,
        const Distribucion& promedio) {
    for(unsigned int i=0;i<ciudadanos->size();i++)
    {
        ciudadanos->at(i).diferencia_ = ciudadanos->at(i).Diferencia(promedio);
    }
    return 0;
}


int AnalizadorDeDatosSecuencial::RegresionLineal(
            const std::unique_ptr<std::map<int,Distribucion> >& ciudadanos) {

  return -3;
}

bool OrdenarPorResiduo (map<int,Distribucion>::const_iterator a,
                        map<int,Distribucion>::const_iterator b) {
    return (a->second.residuo_ < b-> second.residuo_);
}

int AnalizadorDeDatosSecuencial::OrdenarDistribuciones(
        const map<int,Distribucion>& ciudadanos,
        unique_ptr<vector<map<int,Distribucion>::const_iterator> >& indice) {
    indice.reset(new vector<map<int,Distribucion>::const_iterator>() );
    for(map<int,Distribucion>::const_iterator it = ciudadanos.begin();
                it!= ciudadanos.end();
                it++)
        indice->push_back(it);
    sort(indice->begin(),indice->end(),OrdenarPorResiduo);
    return 0;
}
} // namespace implementacion
} // namespace secuencial
} // namespace clasificador_de_distribuciones

