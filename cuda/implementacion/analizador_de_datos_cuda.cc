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


int AnalizadorDeDatosCuda::RegresionLineal(
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

