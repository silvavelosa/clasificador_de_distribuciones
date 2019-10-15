#include <map>
#include <memory>
#include <vector>
#include <UnitTest++.h>

#include "secuencial/implementacion/analizador_de_datos_secuencial.h"

using std::map;
using std::unique_ptr;
using std::vector;

namespace clasificador_de_distribuciones
{
using namespace componentes_compartidos;
namespace secuencial
{
using namespace implementacion;
namespace test
{
SUITE(AnalizadorDeDatosSecuencialTest)
{
    TEST(OrdenarValidaciones)
    {
        unique_ptr<vector<Validacion> > validaciones(new vector<Validacion>());
        Validacion aux;
        aux.id_ciudadano_ = 123456;
        aux.score_ = 123;
        validaciones->push_back(aux);
        aux.id_ciudadano_ = 1234;
        aux.score_ = 256;
        validaciones->push_back(aux);
        aux.id_ciudadano_ = 12345678;
        aux.score_ = 0;
        validaciones->push_back(aux);
        aux.id_ciudadano_ = 123456;
        aux.score_ = 127;
        validaciones->push_back(aux);


        AnalizadorDeDatosSecuencial analizador;
        CHECK_EQUAL(0,analizador.OrdenarValidaciones(validaciones));

        vector<Validacion> esperado;
        aux.id_ciudadano_ = 1234;
        aux.score_ = 256;
        esperado.push_back(aux);
        aux.id_ciudadano_ = 123456;
        aux.score_ = 123;
        esperado.push_back(aux);
        aux.id_ciudadano_ = 123456;
        aux.score_ = 127;
        esperado.push_back(aux);
        aux.id_ciudadano_ = 12345678;
        aux.score_ = 0;
        esperado.push_back(aux);

        CHECK_ARRAY_EQUAL(esperado, *validaciones, esperado.size());
    }
    TEST(AgruparYPromediar)
    {
        vector<Validacion> validaciones;
        Validacion aux;
        aux.id_ciudadano_ = 1234;
        aux.score_ = 0;
        validaciones.push_back(aux);
        aux.score_ = 123;
        validaciones.push_back(aux);
        aux.score_ = 127;
        validaciones.push_back(aux);
        aux.score_ = 129;
        validaciones.push_back(aux);
        aux.score_ = 200;
        validaciones.push_back(aux);
        aux.score_ = 390;
        validaciones.push_back(aux);
        aux.score_ = 500;
        validaciones.push_back(aux);
        aux.score_ = 800;
        validaciones.push_back(aux);
        aux.id_ciudadano_ = 123456;
        aux.score_ = 100;
        validaciones.push_back(aux);
        aux.score_ = 200;
        validaciones.push_back(aux);
        aux.score_ = 256;
        validaciones.push_back(aux);
        aux.score_ = 300;
        validaciones.push_back(aux);
        aux.id_ciudadano_ = 12345678;
        aux.score_ = 0;
        validaciones.push_back(aux);

        AnalizadorDeDatosSecuencial analizador;
        unique_ptr<map<int,Distribucion> > ciudadanos;
        unique_ptr<Distribucion> promedio;
        int est = analizador.AgruparYPromediar(validaciones, ciudadanos, promedio);
        CHECK_EQUAL(0,est);
        map<int,Distribucion>::iterator it = ciudadanos->begin();
        vector<int> esperados[] = {{1,0,0,0,0,0,0,0,0,0,
                                    0,0,3,0,0,0,0,0,0,0,
                                    1,0,0,0,0,0,0,0,0,0,
                                    0,0,0,0,0,0,0,0,0,3},
                                   {0,0,0,0,0,0,0,0,0,0,
                                    1,0,0,0,0,0,0,0,0,0,
                                    1,0,0,0,0,1,0,0,0,0,
                                    1,0,0,0,0,0,0,0,0,0},
                                   {1,0,0,0,0,0,0,0,0,0,
                                    0,0,0,0,0,0,0,0,0,0,
                                    0,0,0,0,0,0,0,0,0,0,
                                    0,0,0,0,0,0,0,0,0,0},
                                   {2,0,0,0,0,0,0,0,0,0,
                                    1,0,3,0,0,0,0,0,0,0,
                                    2,0,0,0,0,1,0,0,0,0,
                                    1,0,0,0,0,0,0,0,0,3}};
        vector<int> total_esperado ={8,4,1,13};
        for(int i=0;i<3;i++)
        {
            CHECK_ARRAY_EQUAL(esperados[i],it->second.frecuencias_,
                               Distribucion::tamano_frecuencias_);
            CHECK_EQUAL(total_esperado[i], it->second.total_);
            it++;
        }
        CHECK_ARRAY_EQUAL(esperados[3],promedio->frecuencias_,
                          Distribucion::tamano_frecuencias_);
        CHECK_EQUAL(total_esperado[3], promedio->total_);

    }

    TEST(OrdenarDistribuciones)
    {
        map<int,Distribucion> ciudadanos;
        unique_ptr<vector<map<int,Distribucion>::const_iterator> > indice;
        Distribucion dist;
        dist.residuo_ = 0.05;
        ciudadanos[1] = dist;
        dist.residuo_ = 0.0001;
        ciudadanos[2] = dist;
        dist.residuo_ = 0.3;
        ciudadanos[3] = dist;
        AnalizadorDeDatosSecuencial analizador;
        analizador.OrdenarDistribuciones(ciudadanos,indice);
        CHECK_EQUAL(3U,indice->size());
        CHECK_EQUAL(2,(*indice)[0]->first);
        CHECK_EQUAL(1,(*indice)[1]->first);
        CHECK_EQUAL(3,(*indice)[2]->first);
    }
}
} // namespace test
} // namespace secuencial
} // namespace clasificador_de_distribuciones
