#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <UnitTest++.h>

#include "open_mp/implementacion/manejador_de_archivos_open_mp.h"

using std::ifstream;
using std::map;
using std::string;
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
SUITE(ManejadorDeArchivosOpenMPTest)
{
    void VerificarArchivo (const string& archivo, const vector<string>& esperado)
    {
            ifstream lectura (archivo);
            REQUIRE CHECK(lectura.is_open());
            string linea;
            for(size_t i=0;i<esperado.size();i++)
            {
                CHECK(getline(lectura, linea));
                CHECK_EQUAL(esperado[i],linea);
            }
            CHECK(!getline(lectura, linea));
            lectura.close();
    }
    TEST(CargarDatosOK)
    {
        string archivo = "test/data/entradaPruebaOK.csv";
        ManejadorDeArchivosOpenMP manejador;
        unique_ptr<vector<Evento> > eventos;
        string msg;
        int res = manejador.CargarDatos(archivo,eventos,msg);
        REQUIRE CHECK_EQUAL(0,res);
        CHECK_EQUAL(5U,eventos->size());
    }
    TEST(CargarDatosGrande)
    {
        string archivo = "test/data/entradaPruebaGrande.csv";
        ManejadorDeArchivosOpenMP manejador;
        unique_ptr<vector<Evento> > eventos;
        string msg;
        int res = manejador.CargarDatos(archivo,eventos,msg);
        REQUIRE CHECK_EQUAL(0,res);
        CHECK_EQUAL(1600U,eventos->size());
    }

    TEST(CargarDatosNOK)
    {
        string archivo = "test/data/entradaPruebaNOK.csv";
        ManejadorDeArchivosOpenMP manejador;
        unique_ptr<vector<Evento> > eventos;
        string msg;
        int res = manejador.CargarDatos(archivo,eventos,msg);
        CHECK_EQUAL(-2,res);
        CHECK_EQUAL("Caracter invalido en la linea 4", msg);
    }

    TEST(CargarDatosNoExiste)
    {
        string archivo = "test/data/archivoQueNoExiste.csv";
        ManejadorDeArchivosOpenMP manejador;
        unique_ptr<vector<Evento> > eventos;
        string msg;
        int res = manejador.CargarDatos(archivo,eventos,msg);
        CHECK_EQUAL(-1,res);
    }

    TEST (GenerarSalida)
    {
        remove( "bin/Test/salidaPrueba.csv" );
        vector<Distribucion> ciudadanos;
        ciudadanos.push_back(Distribucion(1));
        ciudadanos.push_back(Distribucion(7));
        ciudadanos.push_back(Distribucion(3));
        ciudadanos[0].EstablecerResiduo(1.2);
        ciudadanos[1].EstablecerResiduo(0.5);
        ciudadanos[2].EstablecerResiduo(0.003);

        string archivo = "bin/Test/salidaPrueba.csv";
        ManejadorDeArchivosOpenMP manejador;

        string msg;
        int stat = manejador.GenerarSalida(archivo, ciudadanos, msg,
                        IManejadorDeArchivos::ModoDeEscritura::mantener);

        CHECK_EQUAL(0, stat);
        VerificarArchivo("bin/Test/salidaPrueba.csv", {"1;1.2","7;0.5","3;0.003"});

        stat = manejador.GenerarSalida(archivo, ciudadanos, msg,
                        IManejadorDeArchivos::ModoDeEscritura::mantener);

        CHECK_EQUAL(-2, stat);

        stat = manejador.GenerarSalida(archivo, ciudadanos, msg,
                        IManejadorDeArchivos::ModoDeEscritura::concatenar);


        CHECK_EQUAL(0, stat);

        VerificarArchivo("bin/Test/salidaPrueba.csv", {"1;1.2","7;0.5","3;0.003",
                                    "1;1.2","7;0.5","3;0.003"});

        stat = manejador.GenerarSalida(archivo, ciudadanos, msg,
                        IManejadorDeArchivos::ModoDeEscritura::reemplazar);

        CHECK_EQUAL(0, stat);
        VerificarArchivo("bin/Test/salidaPrueba.csv", {"1;1.2","7;0.5","3;0.003"});

        remove( "bin/Test/salidaPrueba.csv" );
    }
}
} // namespace test
} // namespace secuencial
} // namespace clasificador_de_distribuciones

