#include "open_mp/implementacion/analizador_de_datos_open_mp.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <queue>
#include <sstream>

#include <gsl/gsl_multifit.h>
#include <omp.h>

namespace clasificador_de_distribuciones
{
namespace open_mp
{
namespace implementacion
{
using std::map;
using std::pair;
using std::priority_queue;
using std::sort;
using std::string;
using std::unique_ptr;
using std::vector;

class Candidato
{
 public:
    unsigned int posicion_;
    unsigned int origen_;
    Candidato (int posicion, int origen)
    {
        posicion_ = posicion;
        origen_ = origen;
    }
};

template<typename T, class Compare>
class MayorCandidato
{
 public:
  MayorCandidato(const vector<T>& arreglo, const bool& revertir=false)
            : comparar_(), arreglo_(arreglo)
    {
        revertir_=revertir;
    }
  bool operator() (const Candidato& lhs, const Candidato& rhs) const
  {
    if (comparar_(arreglo_[rhs.posicion_],arreglo_[lhs.posicion_]))
        return !revertir_;
    if (comparar_(arreglo_[lhs.posicion_],arreglo_[rhs.posicion_]))
        return revertir_;
    return revertir_^(lhs.origen_ > rhs.origen_);
  }
 private:
    bool revertir_;
    Compare comparar_;
    const vector<T>& arreglo_;
};

template<typename T, class Compare = std::less<T> >
int OrdenamientoParalelo(unique_ptr<vector<T> >& entrada, unsigned int n_hilos )
{
    if(!entrada) return -1;
    if(entrada->size() < 10000)
    {
        sort(entrada->begin(), entrada->end(), Compare());
        return 0;
    }

    unique_ptr<vector<T> > ret(new vector<T>());
    unsigned int fraccion = entrada->size()/n_hilos;

    #pragma omp parallel
    {
        #pragma omp single
        ret->reserve(entrada->size());
        #pragma omp for schedule(dynamic, 1)
        for(unsigned int i=0;i<n_hilos;i++)
        {
            if(i+1<n_hilos)
                sort(entrada->begin()+(i*fraccion),
                     entrada->begin()+((i+1)*fraccion),
                     Compare());
            else
                sort(entrada->begin()+(i*fraccion), entrada->end(), Compare());
        }
    }


    priority_queue< Candidato, vector<Candidato>,
                MayorCandidato<T,Compare> >candidatos(*entrada);
    for(unsigned int i=0;i<n_hilos;i++)
    {
        candidatos.push(Candidato(i*fraccion,i));
    }
    unsigned int origen;
    unsigned int posicion;
    for(unsigned int i=0;i<entrada->size();i++)
    {
        posicion = candidatos.top().posicion_;
        origen = candidatos.top().origen_;
        candidatos.pop();
        ret->push_back(std::move((*entrada)[posicion]));
        if((origen+1U < n_hilos && posicion+1U < (origen+1U)*fraccion) ||
           (origen+1U == n_hilos && posicion+1U < entrada->size()))
        {
            candidatos.push(Candidato(posicion+1,origen));
        }
    }
    entrada = std::move(ret);
    return 0;
}

int AnalizadorDeDatosOpenMP::OrdenarEventos(
        unique_ptr<vector<Evento> >& eventos) {

    return OrdenamientoParalelo<Evento>(eventos, n_hilos_);
}

int AnalizadorDeDatosOpenMP::AgruparYPromediar(
        const vector<Evento>& eventos,
        unique_ptr<vector<Distribucion> >& grupos,
        unique_ptr<Distribucion>& promedio) {

    map<int,int> indice;
    vector<int> actual(n_hilos_,-1);

    promedio.reset(new Distribucion());
    grupos.reset(new vector<Distribucion> ());

    #pragma omp for schedule(dynamic, 100)
    for(unsigned int i=0;i<eventos.size();i++)
    {
        int id_hilo = omp_get_thread_num();
        if(actual[id_hilo] == -1 ||
           (*grupos)[actual[id_hilo]].Grupo() != eventos[i].id_grupo_)
        {
            #pragma omp critical
            {
                map<int,int>::iterator it = indice.find(eventos[i].id_grupo_);
                if(it == indice.end())
                {
                    grupos->push_back(Distribucion(eventos[i].id_grupo_));
                    indice[eventos[i].id_grupo_] = grupos->size()-1;
                    actual[id_hilo] = grupos->size()-1;
                }
                else
                {
                    actual[id_hilo] = it->second;
                }
            }
        }
        (*grupos)[actual[id_hilo]].AnadirEvento(eventos[i]);
        promedio->AnadirEvento(eventos[i]);
    }
    return 0;
}

int AnalizadorDeDatosOpenMP::CompararDistribuciones(
        vector<Distribucion>& grupos,
        const Distribucion& promedio) {

    #pragma omp for schedule(dynamic, 10)
    for(unsigned int i=0;i<grupos.size();i++)
    {
        grupos[i].EstablecerDiferencia(promedio);
    }
    return 0;
}


int AnalizadorDeDatosOpenMP::RegresionLineal(
            vector<Distribucion>& grupos) {

  return -3;
}

class OrdenarPorResiduo
{
  bool revertir_;
public:
  OrdenarPorResiduo(const bool& revertir=false)
    {revertir_=revertir;}
  bool operator() (const Distribucion& lhs, const Distribucion& rhs) const
  {
    return revertir_^(lhs.Residuo() < rhs.Residuo());
  }
};

int AnalizadorDeDatosOpenMP::OrdenarDistribuciones(
        unique_ptr<vector<Distribucion> >& grupos) {
    return OrdenamientoParalelo<Distribucion,OrdenarPorResiduo>(grupos, n_hilos_);
}
} // namespace implementacion
} // namespace open_mp
} // namespace clasificador_de_distribuciones

