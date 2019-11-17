#include "open_mp/implementacion/manejador_de_archivos_open_mp.h"

#include <fstream>
#include <iostream>
#include <sstream>

#include "secuencial/implementacion/manejador_de_archivos_secuencial.h"

namespace clasificador_de_distribuciones
{
namespace open_mp
{
namespace implementacion
{
using std::ios;
using std::map;
using std::max;
using std::min;
using std::string;
using std::unique_ptr;
using std::vector;

int ManejadorDeArchivosOpenMP::CargarDatos(const string& archivo,
        unique_ptr<vector<Evento> >& eventos,
        string& msg) {
    omp_set_num_threads(n_hilos_);
    int tamano = TamanoDeArchivo(archivo);

    if(tamano < 10000000)
    {
        secuencial::implementacion::ManejadorDeArchivosSecuencial manejador_sec;
        return manejador_sec.CargarDatos(archivo, eventos, msg);
    }


    std::ifstream entrada(archivo);
    if(!entrada.is_open())
    {
        eventos.reset(nullptr);
        return -1;
    }

    eventos.reset(new vector<Evento>);
    eventos->resize(tamano/5);

    char* contenido = (char*) malloc(tamano*sizeof(char) + 100);

    entrada.read(contenido, tamano + 100);
    size_t n_caracteres = entrada.gcount();

    for(size_t i = n_caracteres; i< tamano + 100; i++)
        contenido[i] = '\0';

    size_t saltar = n_caracteres/n_hilos_;
    size_t evento_actual = 0;
    size_t tamano_real = 0;

    int ret =0;

    #pragma omp parallel
    {
        unsigned int i = omp_get_thread_num();
        size_t caracter = 0;

        if(i > 0)
        {
            for(caracter = i*saltar;
                contenido[caracter] != '\n' && contenido[caracter] != '\0';
                caracter++);
            caracter++;
        }

        size_t pos=100, ini=0, fin = 100;
        size_t avance;
        while((caracter <= (i+1)* saltar || i == n_hilos_-1) && ret == 0 &&
            contenido[caracter] != '\0')
        {
            if(pos == fin)
            {
                #pragma omp critical
                {
                    pos = ini = evento_actual;
                    evento_actual+= max(1, (int) (((i+1)*saltar)-caracter)/25);
                    fin = evento_actual;
                }
            }
            int est =
                Evento::Parse(contenido + caracter,';', (*eventos)[pos],&avance);
            pos++;
            if(est != Evento::ParseResult::OK)
            {
                std::stringstream ss;
                switch(est)
                {
                case Evento::ParseResult::IdGrupoVacio:
                    ss<<"IdGrupo vacio en la linea "<<i;
                    break;
                case Evento::ParseResult::ValorVacio:
                    ss<<"Valor vacio en la linea "<<i;
                    break;
                case Evento::ParseResult::CaracterInvalido:
                    ss<<"Caracter invalido en la linea "<<i;
                    break;
                case Evento::ParseResult::LineaIncompleta:
                    ss<<"Linea incompleta "<<i;
                    break;
                case Evento::ParseResult::DatosSobrantes:
                    ss<<"Datos sobrantes en la linea "<<i<<entrada.tellg();
                    break;
                }
                msg = ss.str();
                ret = -2;
            }
            caracter+=avance+1;
        }
        #pragma omp critical
        {
            tamano_real= max(tamano_real, pos);
        }
    }

    entrada.close();
    if(ret < 0)
    {
        eventos.reset(nullptr);
        return ret;
    }

    eventos->resize(tamano_real);
    eventos->shrink_to_fit();
    return 0;
}

int ManejadorDeArchivosOpenMP::GenerarSalida(const string& archivo,
        const vector<Distribucion>& grupos,
        string& msg,
        IManejadorDeArchivos::ModoDeEscritura modo) {

    secuencial::implementacion::ManejadorDeArchivosSecuencial manejador_sec;
    return manejador_sec.GenerarSalida(archivo, grupos, msg, modo);
}

int ManejadorDeArchivosOpenMP::TamanoDeArchivo(const string& archivo)
{

    std::ifstream entrada (archivo, std::ios::binary);
    if(!entrada.is_open())
    {
        return -1;
    }
    std::streampos ini = entrada.tellg();
    entrada.seekg (0, std::ios::end);
    std::streampos fin = entrada.tellg();
    entrada.close();
    return fin-ini;
}
} // namespace implementacion
} // namespace open_mp
} // namespace clasificador_de_distribuciones
