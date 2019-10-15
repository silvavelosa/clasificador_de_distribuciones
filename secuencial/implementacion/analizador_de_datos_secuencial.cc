#include "secuencial/implementacion/analizador_de_datos_secuencial.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>

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

int AnalizadorDeDatosSecuencial::OrdenarValidaciones(
        unique_ptr<vector<Validacion> >& validaciones) {
    sort(validaciones->begin(), validaciones->end());
    return 0;
}

int AnalizadorDeDatosSecuencial::AgruparYPromediar(
        const vector<Validacion>& validaciones,
        unique_ptr<map<int,Distribucion> >& ciudadanos,
        unique_ptr<Distribucion>& promedio) {
    promedio.reset(new Distribucion());
    ciudadanos.reset(new map<int,Distribucion> ());
    map<int,Distribucion>::iterator actual = ciudadanos->end();
    for(int i=0;i<validaciones.size();i++)
    {
        if(actual == ciudadanos->end()
                || actual->first != validaciones[i].id_ciudadano_)
        {
            actual = ciudadanos->emplace_hint(ciudadanos->end(),
                                      validaciones[i].id_ciudadano_,Distribucion());
        }
        if(Distribucion::tamano_frecuencias_ <= validaciones[i].score_/10)
        {
            actual->second.frecuencias_[Distribucion::tamano_frecuencias_-1]++;
            promedio->frecuencias_[Distribucion::tamano_frecuencias_-1]++;
        }
        else
        {
            actual->second.frecuencias_[validaciones[i].score_/10]++;
            promedio->frecuencias_[validaciones[i].score_/10]++;
        }
        actual->second.total_++;
        promedio->total_++;
    }
    return 0;
}

int AnalizadorDeDatosSecuencial::CompararDistribuciones(
        const unique_ptr<map<int,Distribucion> >& ciudadanos,
        const Distribucion& promedio) {
    return -3;
}

int AnalizadorDeDatosSecuencial::OrdenarDistribuciones(
        const map<int,Distribucion>& ciudadanos,
        unique_ptr<vector<map<int,Distribucion>::iterator> >& indice) {
    return -3;
}

} // namespace implementacion
} // namespace secuencial
} // namespace clasificador_de_distribuciones

