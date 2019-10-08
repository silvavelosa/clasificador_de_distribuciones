#include "secuencial/implementacion/analizador_de_datos_secuencial.h"

#include <cstdlib>
#include <fstream>
#include <sstream>

namespace clasificador_de_distribuciones
{
namespace secuencial
{
namespace implementacion
{
using std::map;
using std::string;
using std::unique_ptr;
using std::vector;

int AnalizadorDeDatosSecuencial::OrdenarValidaciones(
        unique_ptr<vector<Validacion> >& validaciones) {
    return -3;
}

int AnalizadorDeDatosSecuencial::AgruparYPromediar(
        const vector<Validacion >& valideaciones,
        unique_ptr<map<int,Distribucion> >& ciudadanos,
        Distribucion& promedio) {
    return -3;
}

int AnalizadorDeDatosSecuencial::CompararDistribuciones(
        const unique_ptr<map<int,Distribucion> >& ciudadanos,
        const Distribucion& promedio) {
    return -3;
}

int AnalizadorDeDatosSecuencial::OrdenarDistribuciones(
        const map<int,Distribucion>& ciudadanos,
        unique_ptr<vector<map<int,Distribucion>::iterator> >& indice) {
    return -3;
}

} // namespace implementacion
} // namespace secuencial
} // namespace clasificador_de_distribuciones

