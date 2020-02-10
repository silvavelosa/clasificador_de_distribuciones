#include "open_mp/implementacion/manejador_de_archivos_open_mp.h"

#include <fstream>
#include <iostream>
#include <sstream>

#include "componentes_compartidos/util_archivos.h"

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

    eventos.reset(new vector<Evento>);
    eventos->resize(tamano/5);

    size_t saltar = tamano/n_hilos_;
    size_t evento_actual = 0;
    size_t tamano_real = 0;

    int ret =0;

    #pragma omp parallel
    {
        char* contenido = (char*) malloc(saltar*sizeof(char) + 100);
        std::ifstream entrada(archivo, ios::binary | ios::in);
        if(!entrada.is_open())
        {
            ret = -1;
        }

        unsigned int i = omp_get_thread_num();


        entrada.seekg(saltar*i, ios::beg);
        entrada.read(contenido, saltar + 100);

        if(entrada.eof())
        {
            for(size_t j = entrada.gcount(); j< saltar + 100; j++)
                contenido[j] = '\0';
        }
        else
            contenido[saltar+99] = '\0';

        entrada.close();

        size_t caracter = 0;
        if(i > 0)
        {
            for(;contenido[caracter] != '\n' && contenido[caracter] != '\r'
                    && contenido[caracter] != '\0';
                caracter++);
            if(contenido[caracter] == '\r' && contenido[caracter+1] == '\n')
                caracter++;
            caracter++;
        }

        size_t pos=100, ini=0, fin = 100;
        size_t avance;
        while((caracter <= saltar || i == n_hilos_-1) && ret == 0 &&
            contenido[caracter] != '\0')
        {
            if(pos == fin)
            {
                #pragma omp critical
                {
                    pos = ini = evento_actual;
                    evento_actual+= max(1, (int) ((saltar-caracter)/25));
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
                    ss<<"IdGrupo vacio en la linea "<<caracter+(i*saltar);
                    break;
                case Evento::ParseResult::ValorVacio:
                    ss<<"Valor vacio en la linea "<<caracter+(i*saltar);
                    break;
                case Evento::ParseResult::CaracterInvalido:
                    ss<<"Caracter invalido en la linea "<<caracter+(i*saltar);
                    break;
                case Evento::ParseResult::LineaIncompleta:
                    ss<<"Linea incompleta "<<caracter+(i*saltar);
                    break;
                case Evento::ParseResult::DatosSobrantes:
                    ss<<"Datos sobrantes en la linea "<<caracter+(i*saltar);
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
        free(contenido);
    }

    if(ret < 0)
    {
        eventos.reset(nullptr);
        return ret;
    }

    eventos->resize(tamano_real);
    eventos->shrink_to_fit();
    return 0;
}
} // namespace implementacion
} // namespace open_mp
} // namespace clasificador_de_distribuciones
