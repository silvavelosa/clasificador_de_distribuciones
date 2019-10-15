#include <memory>
#include <string>
#include <vector>
#include <UnitTest++.h>

#include "componentes_compartidos/entidades.h"
#include "secuencial/implementacion/manejador_de_archivos_secuencial.h"

using std::string;
using std::unique_ptr;
using std::vector;

using clasificador_de_distribuciones::secuencial::implementacion::ManejadorDeArchivosSecuencial;

namespace clasificador_de_distribuciones
{
using namespace componentes_compartidos;
namespace secuencial
{
namespace test
{
SUITE(ManejadorDeArchivosSecuencialTest)
{
    TEST(CargarDatosOK)
    {
        string archivo = "test/entradaPruebaOK.csv";
        ManejadorDeArchivosSecuencial manejador;
        unique_ptr<vector<Validacion> > validaciones;
        string msg;
        int res = manejador.CargarDatos(archivo,validaciones,msg);
        REQUIRE CHECK_EQUAL(0,res);
        CHECK(validaciones->size() == 3);
    }

    TEST(CargarDatosNOK)
    {
        string archivo = "test/entradaPruebaNOK.csv";
        ManejadorDeArchivosSecuencial manejador;
        unique_ptr<vector<Validacion> > validaciones;
        string msg;
        int res = manejador.CargarDatos(archivo,validaciones,msg);
        CHECK_EQUAL(-2,res);
        CHECK_EQUAL("Caracter invalido en la linea 4", msg);
    }

    TEST(CargarDatosNoExiste)
    {
        string archivo = "test/archivoQueNoExiste.csv";
        ManejadorDeArchivosSecuencial manejador;
        unique_ptr<vector<Validacion> > validaciones;
        string msg;
        int res = manejador.CargarDatos(archivo,validaciones,msg);
        CHECK_EQUAL(-1,res);
    }
}
} // namespace test
} // namespace secuencial
} // namespace clasificador_de_distribuciones

