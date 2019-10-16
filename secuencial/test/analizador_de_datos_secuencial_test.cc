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
    TEST(OrdenarEventos)
    {
        unique_ptr<vector<Evento> > eventos(new vector<Evento>());
        Evento aux;
        aux.id_grupo_ = 123456;
        aux.valor_ = 123;
        eventos->push_back(aux);
        aux.id_grupo_ = 1234;
        aux.valor_ = 256;
        eventos->push_back(aux);
        aux.id_grupo_ = 12345678;
        aux.valor_ = 0;
        eventos->push_back(aux);
        aux.id_grupo_ = 123456;
        aux.valor_ = 127;
        eventos->push_back(aux);


        AnalizadorDeDatosSecuencial analizador;
        CHECK_EQUAL(0,analizador.OrdenarEventos(eventos));

        vector<Evento> esperado;
        aux.id_grupo_ = 1234;
        aux.valor_ = 256;
        esperado.push_back(aux);
        aux.id_grupo_ = 123456;
        aux.valor_ = 123;
        esperado.push_back(aux);
        aux.id_grupo_ = 123456;
        aux.valor_ = 127;
        esperado.push_back(aux);
        aux.id_grupo_ = 12345678;
        aux.valor_ = 0;
        esperado.push_back(aux);

        CHECK_ARRAY_EQUAL(esperado, *eventos, esperado.size());
    }
    TEST(AgruparYPromediar)
    {
        vector<Evento> eventos;
        Evento aux;
        aux.id_grupo_ = 1234;
        aux.valor_ = 0;
        eventos.push_back(aux);
        aux.valor_ = 123;
        eventos.push_back(aux);
        aux.valor_ = 127;
        eventos.push_back(aux);
        aux.valor_ = 129;
        eventos.push_back(aux);
        aux.valor_ = 200;
        eventos.push_back(aux);
        aux.valor_ = 390;
        eventos.push_back(aux);
        aux.valor_ = 500;
        eventos.push_back(aux);
        aux.valor_ = 800;
        eventos.push_back(aux);
        aux.id_grupo_ = 123456;
        aux.valor_ = 100;
        eventos.push_back(aux);
        aux.valor_ = 200;
        eventos.push_back(aux);
        aux.valor_ = 256;
        eventos.push_back(aux);
        aux.valor_ = 300;
        eventos.push_back(aux);
        aux.id_grupo_ = 12345678;
        aux.valor_ = 0;
        eventos.push_back(aux);

        AnalizadorDeDatosSecuencial analizador;
        unique_ptr<map<int,Distribucion> > ciudadanos;
        unique_ptr<Distribucion> promedio;
        int est = analizador.AgruparYPromediar(eventos, ciudadanos, promedio);
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