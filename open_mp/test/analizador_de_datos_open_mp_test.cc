#include <algorithm>
#include <map>
#include <memory>
#include <vector>
#include <UnitTest++.h>

#include "open_mp/implementacion/analizador_de_datos_open_mp.h"

using std::map;
using std::unique_ptr;
using std::vector;

namespace clasificador_de_distribuciones
{
using namespace componentes_compartidos;
namespace open_mp
{
using namespace implementacion;
namespace test
{
SUITE(AnalizadorDeDatosOpenMPTest)
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


        AnalizadorDeDatosOpenMP analizador;
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
    TEST (OrdenarEventosPar)
    {
        unique_ptr<vector<Evento> > eventos(new vector<Evento>());
        vector<Evento> esperado;
        eventos->reserve(10000);
        esperado.reserve(10000);
        for(int i=0;i<10000;i++)
        {
            Evento x;
            x.id_grupo_ =10001- i;
            x.valor_ = 100;
            eventos->push_back(x);
            esperado.push_back(x);
        }
        for(int i=0;i<10000;i++)
        {
            Evento x;
            x.id_grupo_ = 1;
            x.valor_ = 10001 - i;
            eventos->push_back(x);
            esperado.push_back(x);
        }

        std::sort(esperado.begin(),esperado.end());

        AnalizadorDeDatosOpenMP analizador;
        CHECK_EQUAL(0,analizador.OrdenarEventos(eventos));
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

        AnalizadorDeDatosOpenMP analizador;
        unique_ptr<vector<Distribucion> > ciudadanos;
        unique_ptr<Distribucion> promedio;
        int est = analizador.AgruparYPromediar(eventos, ciudadanos, promedio);
        CHECK_EQUAL(0,est);
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
            CHECK_ARRAY_EQUAL(esperados[i],(*ciudadanos)[i].Frecuencias(),
                               Distribucion::tamano_frecuencias_);
            CHECK_EQUAL(total_esperado[i], (*ciudadanos)[i].Total());
        }
        CHECK_ARRAY_EQUAL(esperados[3],promedio->Frecuencias(),
                          Distribucion::tamano_frecuencias_);
        CHECK_EQUAL(total_esperado[3], promedio->Total());

    }

    TEST(OrdenarDistribuciones)
    {
        unique_ptr<vector<Distribucion> >ciudadanos(new vector<Distribucion>());
        ciudadanos->push_back(Distribucion(1));
        ciudadanos->back().EstablecerResiduo(0.05);
        ciudadanos->push_back(Distribucion(2));
        ciudadanos->back().EstablecerResiduo(0.0001);
        ciudadanos->push_back(Distribucion(3));
        ciudadanos->back().EstablecerResiduo(0.3);

        AnalizadorDeDatosOpenMP analizador;
        analizador.OrdenarDistribuciones(ciudadanos);
        CHECK_EQUAL(2,(*ciudadanos)[0].Grupo());
        CHECK_EQUAL(1,(*ciudadanos)[1].Grupo());
        CHECK_EQUAL(3,(*ciudadanos)[2].Grupo());
    }
}
} // namespace test
} // namespace secuencial
} // namespace clasificador_de_distribuciones
