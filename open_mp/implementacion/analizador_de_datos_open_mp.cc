#include "open_mp/implementacion/analizador_de_datos_open_mp.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <queue>
#include <sstream>

#include <gsl/gsl_multifit.h>

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

vector<Distribucion> merge (vector<Distribucion>& a,
                                vector <Distribucion>& b)
{
    vector<Distribucion> ret;
    ret.reserve(a.size() + b.size());
    size_t l,r;
    l=r=0;
    while( l < a.size() || r < b.size() )
    {
        if ( l == a.size() || (r < b.size() && b[r].Grupo() < a[l].Grupo() ) )
        {
            ret.push_back( std::move( b[r] ) );
            r++;
        }
        else if ( r == b.size() || a[l].Grupo() < b[r].Grupo() )
        {
            ret.push_back( std::move( a[l] ) );
            l++;
        }
        else
        {
            ret.push_back( std::move( a[l] ) );
            ret.back() += b[r];
            r++;
            l++;
        }
    }
    ret.shrink_to_fit();
    return ret;
}

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
    size_t fraccion = entrada->size()/n_hilos;
    vector<size_t> limites(n_hilos);

    #pragma omp parallel
    {
        #pragma omp single nowait
        ret->resize(entrada->size());

        size_t i = omp_get_thread_num();
        limites[i] = i*fraccion;

        if(i < n_hilos-1)
            sort(entrada->begin() + (i*fraccion),
                 entrada->begin() + ((i+1)*fraccion),
                 Compare());
        else
            sort(entrada->begin() + (i*fraccion),
                 entrada->end(),
                 Compare());
    }


    Compare comparar;
    size_t incremento = 1;
    while(incremento < n_hilos)
    {
        #pragma omp parallel for schedule(dynamic, 1)
        for(size_t i=0;i<n_hilos;i+=incremento*2)
        {
            if(i+incremento >= n_hilos)
            {
                for(size_t j=limites[i]; j< entrada->size();j++)
                    (*ret)[j] = std::move((*entrada)[j]);
            }
            else
            {
                size_t izq = limites[i];
                size_t der = limites[i+incremento];
                size_t fin = entrada->size();
                if(i+(incremento*2) < n_hilos)
                    fin = limites[i+(incremento*2)];

                for(size_t j=limites[i]; j<fin; j++)
                {
                    if(izq == limites[i+incremento] ||
                       (der < fin && comparar((*entrada)[der],(*entrada)[izq])))
                    {
                        (*ret)[j] = std::move((*entrada)[der]);
                        der++;
                    }
                    else
                    {
                        (*ret)[j] = std::move((*entrada)[izq]);
                        izq++;
                    }
                }
            }
        }

        swap(entrada,ret);
        incremento*=2;
    }

    return 0;
}

int AnalizadorDeDatosOpenMP::OrdenarEventos(
        unique_ptr<vector<Evento> >& eventos) {
    omp_set_num_threads(n_hilos_);
    return OrdenamientoParalelo<Evento>(eventos, n_hilos_);
}

int AnalizadorDeDatosOpenMP::AgruparYPromediar(
        const vector<Evento>& eventos,
        unique_ptr<vector<Distribucion> >& grupos,
        unique_ptr<Distribucion>& promedio) {
    omp_set_num_threads(n_hilos_);

    promedio.reset(new Distribucion());
    Distribucion& p = *promedio;
    grupos.reset(new vector<Distribucion> ());
    vector<Distribucion>& g = *grupos;

    #pragma omp declare reduction(+ : Distribucion : omp_out += omp_in ) 
    #pragma omp declare reduction(merge : vector<Distribucion> : \
                    omp_out = merge( omp_out, omp_in )  )
    #pragma omp parallel for reduction(+: p) \
                                reduction(merge: g) \
                                schedule(dynamic, 100000)
    for(size_t i=0;i<eventos.size();i++)
    {
        if(g.empty() || g.back().Grupo() != eventos[i].id_grupo_)
        {
            g.push_back(Distribucion(eventos[i].id_grupo_));
        }
        g.back().AnadirEvento(eventos[i]);
        p.AnadirEvento(eventos[i]);
    }
    return 0;
}

int AnalizadorDeDatosOpenMP::CompararDistribuciones(
        vector<Distribucion>& grupos,
        const Distribucion& promedio) {
    omp_set_num_threads(n_hilos_);
    #pragma omp parallel for schedule(dynamic, 10)
    for(size_t i=0;i<grupos.size();i++)
    {
        grupos[i].EstablecerDiferencia(promedio);
    }
    return 0;
}


int AnalizadorDeDatosOpenMP::RegresionLineal(
            vector<Distribucion>& grupos) {
 size_t n = grupos.size();
    size_t n_mayores_50 = 0U;
    for(size_t i=0;i<n;i++)
    {
        if(grupos[i].Total() > 50)
        {
            n_mayores_50++;
        }
    }
    gsl_matrix* tamanos = gsl_matrix_alloc(n_mayores_50,2);
    gsl_vector* diferencias = gsl_vector_alloc(n_mayores_50);
    gsl_vector* coef = gsl_vector_alloc(2);
    gsl_matrix* cov = gsl_matrix_alloc(2,2);
    for(size_t i=0, j=0;i<n && j< n_mayores_50;i++)
    {
        if(grupos[i].Total() > 50)
        {
            gsl_vector_set(diferencias, j, log(grupos[i].Diferencia()));
            gsl_matrix_set(tamanos,j, 0, 1.0);
            gsl_matrix_set(tamanos,j, 1, log(grupos[i].Total()));
            j++;
        }
    }

    gsl_multifit_robust_workspace* work
        = gsl_multifit_robust_alloc (gsl_multifit_robust_default,n_mayores_50, 2);
    int s;
    s = gsl_multifit_robust (tamanos, diferencias, coef, cov, work);

    if(s == GSL_SUCCESS || s == GSL_EMAXITER)
    {
        gsl_multifit_robust_stats estadisticas = gsl_multifit_robust_statistics(work);
        for(size_t i=0, j=0;i<n && j< n_mayores_50;i++)
        {
            if(grupos[i].Total() > 50)
            {
                grupos[i].EstablecerResiduo(gsl_vector_get(estadisticas.r,j));
                j++;
            }
        }
    }
    else
    {
        return -1;
    }

    gsl_multifit_robust_free (work);
    gsl_matrix_free(cov);
    gsl_matrix_free(tamanos);
    gsl_vector_free(coef);
    gsl_vector_free(diferencias);
    return 0;
}

class OrdenarPorResiduo
{
  bool revertir_;
public:
  OrdenarPorResiduo(const bool& revertir=false)
    {revertir_=revertir;}
  bool operator() (const Distribucion& lhs, const Distribucion& rhs) const
  {
    return revertir_^(lhs.Residuo() > rhs.Residuo());
  }
};

int AnalizadorDeDatosOpenMP::OrdenarDistribuciones(
        unique_ptr<vector<Distribucion> >& grupos) {
    omp_set_num_threads(n_hilos_);
    return OrdenamientoParalelo<Distribucion,OrdenarPorResiduo>(grupos, n_hilos_);
}
} // namespace implementacion
} // namespace open_mp
} // namespace clasificador_de_distribuciones

