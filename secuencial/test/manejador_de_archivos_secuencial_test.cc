#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <UnitTest++.h>

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
            for(unsigned int i=0;i<esperado.size();i++)
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
        unique_ptr<vector<Validacion> > validaciones;
        string msg;
        int res = manejador.CargarDatos(archivo,validaciones,msg);
        REQUIRE CHECK_EQUAL(0,res);
        CHECK(validaciones->size() == 3);
    }

    TEST(CargarDatosNOK)
    {
        string archivo = "test/data/entradaPruebaNOK.csv";
        ManejadorDeArchivosSecuencial manejador;
        unique_ptr<vector<Validacion> > validaciones;
        string msg;
        int res = manejador.CargarDatos(archivo,validaciones,msg);
        CHECK_EQUAL(-2,res);
        CHECK_EQUAL("Caracter invalido en la linea 4", msg);
    }

    TEST(CargarDatosNoExiste)
    {
        string archivo = "test/data/archivoQueNoExiste.csv";
        ManejadorDeArchivosSecuencial manejador;
        unique_ptr<vector<Validacion> > validaciones;
        string msg;
        int res = manejador.CargarDatos(archivo,validaciones,msg);
        CHECK_EQUAL(-1,res);
    }

    TEST (GenerarSalidaMap)
    {
        remove( "bin/Test/salidaPrueba.csv" );
        map<int,Distribucion> ciudadanos;
        ciudadanos[1].diferencia_ = 1.2;
        ciudadanos[7].diferencia_ = 0.5;
        ciudadanos[3].diferencia_ = 0.003;

        string archivo = "bin/Test/salidaPrueba.csv";
        ManejadorDeArchivosSecuencial manejador;

        string msg;
        int stat = manejador.GenerarSalida(archivo, ciudadanos, msg,
                        IManejadorDeArchivos::ModoDeEscritura::mantener);

        CHECK_EQUAL(0, stat);
        VerificarArchivo("bin/Test/salidaPrueba.csv", {"1;1.2","3;0.003","7;0.5"});

        stat = manejador.GenerarSalida(archivo, ciudadanos, msg,
                        IManejadorDeArchivos::ModoDeEscritura::mantener);

        CHECK_EQUAL(-2, stat);

        stat = manejador.GenerarSalida(archivo, ciudadanos, msg,
                        IManejadorDeArchivos::ModoDeEscritura::concatenar);


        CHECK_EQUAL(0, stat);

        VerificarArchivo("bin/Test/salidaPrueba.csv", {"1;1.2","3;0.003","7;0.5",
                                    "1;1.2","3;0.003","7;0.5"});

        stat = manejador.GenerarSalida(archivo, ciudadanos, msg,
                        IManejadorDeArchivos::ModoDeEscritura::reemplazar);

        CHECK_EQUAL(0, stat);
        VerificarArchivo("bin/Test/salidaPrueba.csv", {"1;1.2","3;0.003","7;0.5"});

        remove( "bin/Test/salidaPrueba.csv" );
    }
    TEST (GenerarSalidaIndice)
    {
        remove( "bin/Test/salidaPrueba.csv" );
        map<int,Distribucion> ciudadanos;
        ciudadanos[1].diferencia_ = 1.2;
        ciudadanos[7].diferencia_ = 0.5;
        ciudadanos[3].diferencia_ = 0.003;
        vector<map<int,Distribucion>::const_iterator> indice;
        indice.push_back(ciudadanos.find(1));
        indice.push_back(ciudadanos.find(7));
        indice.push_back(ciudadanos.find(3));

        string archivo = "bin/Test/salidaPrueba.csv";
        ManejadorDeArchivosSecuencial manejador;

        string msg;
        int stat = manejador.GenerarSalida(archivo, indice, msg,
                        IManejadorDeArchivos::ModoDeEscritura::mantener);
        CHECK_EQUAL(0, stat);
        VerificarArchivo("bin/Test/salidaPrueba.csv", {"1;1.2","7;0.5","3;0.003"});

        stat = manejador.GenerarSalida(archivo, indice, msg,
                        IManejadorDeArchivos::ModoDeEscritura::mantener);
        CHECK_EQUAL(-2, stat);

        stat = manejador.GenerarSalida(archivo, indice, msg,
                        IManejadorDeArchivos::ModoDeEscritura::concatenar);
        CHECK_EQUAL(0, stat);
        VerificarArchivo("bin/Test/salidaPrueba.csv", {"1;1.2","7;0.5","3;0.003",
                                    "1;1.2","7;0.5","3;0.003"});

        stat = manejador.GenerarSalida(archivo, indice, msg,
                        IManejadorDeArchivos::ModoDeEscritura::reemplazar);

        CHECK_EQUAL(0, stat);
        VerificarArchivo("bin/Test/salidaPrueba.csv", {"1;1.2","7;0.5","3;0.003"});

        remove( "bin/Test/salidaPrueba.csv" );
    }
}
} // namespace test
} // namespace secuencial
} // namespace clasificador_de_distribuciones

