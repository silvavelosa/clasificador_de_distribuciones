#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <UnitTest++/UnitTest++.h>

#include "secuencial/implementacion/manejador_de_archivos_secuencial.h"

using std::ifstream;
using std::map;
using std::string;
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
SUITE(ManejadorDeArchivosSecuencialTest)
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
        ManejadorDeArchivosSecuencial manejador;
        unique_ptr<vector<Evento> > eventos;
        string msg;
        int res = manejador.CargarDatos(archivo,eventos,msg);
        REQUIRE CHECK_EQUAL(0,res);
        CHECK(eventos->size() == 3);
    }

    TEST(CargarDatosNOK)
    {
        string archivo = "test/data/entradaPruebaNOK.csv";
        ManejadorDeArchivosSecuencial manejador;
        unique_ptr<vector<Evento> > eventos;
        string msg;
        int res = manejador.CargarDatos(archivo,eventos,msg);
        CHECK_EQUAL(-2,res);
        CHECK_EQUAL("Caracter invalido en la linea 4", msg);
    }

    TEST(CargarDatosNoExiste)
    {
        string archivo = "test/data/archivoQueNoExiste.csv";
        ManejadorDeArchivosSecuencial manejador;
        unique_ptr<vector<Evento> > eventos;
        string msg;
        int res = manejador.CargarDatos(archivo,eventos,msg);
        CHECK_EQUAL(-1,res);
    }

    TEST (GenerarSalida)
    {
        remove( "test/data/salidaPrueba.csv" );
        vector<Distribucion> ciudadanos;
        ciudadanos.push_back(Distribucion(1));
        ciudadanos.push_back(Distribucion(7));
        ciudadanos.push_back(Distribucion(3));
        ciudadanos[0].EstablecerResiduo(1.2);
        ciudadanos[1].EstablecerResiduo(0.5);
        ciudadanos[2].EstablecerResiduo(0.003);

        string archivo = "test/data/salidaPrueba.csv";
        ManejadorDeArchivosSecuencial manejador;

        string msg;
        int stat = manejador.GenerarSalida(archivo, ciudadanos, msg,
                        IManejadorDeArchivos::ModoDeEscritura::mantener);

        CHECK_EQUAL(0, stat);
        VerificarArchivo("test/data/salidaPrueba.csv", {"1;1.2","7;0.5","3;0.003"});

        stat = manejador.GenerarSalida(archivo, ciudadanos, msg,
                        IManejadorDeArchivos::ModoDeEscritura::mantener);

        CHECK_EQUAL(-2, stat);

        stat = manejador.GenerarSalida(archivo, ciudadanos, msg,
                        IManejadorDeArchivos::ModoDeEscritura::concatenar);


        CHECK_EQUAL(0, stat);

        VerificarArchivo("test/data/salidaPrueba.csv", {"1;1.2","7;0.5","3;0.003",
                                    "1;1.2","7;0.5","3;0.003"});

        stat = manejador.GenerarSalida(archivo, ciudadanos, msg,
                        IManejadorDeArchivos::ModoDeEscritura::reemplazar);

        CHECK_EQUAL(0, stat);
        VerificarArchivo("test/data/salidaPrueba.csv", {"1;1.2","7;0.5","3;0.003"});

        remove( "test/data/salidaPrueba.csv" );
    }
}
} // namespace test
} // namespace secuencial
} // namespace clasificador_de_distribuciones

