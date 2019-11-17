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

    int tamano = TamanoDeArchivo(archivo);

    if(tamano <= 0)
    {
        eventos.reset(nullptr);
        return -1;
    }

    if(tamano < 10000)
    {
        secuencial::implementacion::ManejadorDeArchivosSecuencial manejador_sec;
        return manejador_sec.CargarDatos(archivo, eventos, msg);
    }

    eventos.reset(new vector<Evento>);
    eventos->resize(tamano/5);
    size_t saltar = tamano/n_hilos_;
    size_t evento_actual = 0, tamano_real = 0;
    int ret =0;


    #pragma omp parallel for schedule(dynamic, 1)
    for(unsigned int i=0;i<n_hilos_;i++)
    {
        size_t caracter = 0;
        std::ifstream entrada(archivo);
        char linea[25];

        if(!entrada.is_open())
        {
            ret = -1;
        }

        entrada.seekg(saltar*i, ios::beg);

        if(i!=0)
        {
            entrada.getline(linea, 25);
            caracter += entrada.gcount();
        }

        size_t pos=100, ini=0, fin = 100;

        while((caracter <= saltar || i == n_hilos_-1) && ret == 0 &&
            entrada.getline(linea, 25))
        {
            if(pos == fin)
            {
                #pragma omp critical
                {
                    pos = ini = evento_actual;
                    evento_actual+= max(1, (int) (saltar-caracter)/25);
                    fin = evento_actual;
                }
            }
            int est = Evento::Parse(linea, (*eventos)[pos]);
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

            caracter += entrada.gcount();
        }
        #pragma omp critical
        {
            /*std::cout<<"POS:"<<pos<<" "<<i<<std::endl;
            std::cout<<entrada.tellg()<<" "<<saltar*(i+1)<<std::endl;*/
            tamano_real= max(tamano_real, pos);
        }
        entrada.close();
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

int ManejadorDeArchivosOpenMP::GenerarSalida(const string& archivo,
        const vector<Distribucion>& grupos,
        string& msg,
        IManejadorDeArchivos::ModoDeEscritura modo) {

    int tamano = TamanoDeArchivo(archivo);

    if(tamano > 0 && modo == ModoDeEscritura::mantener)
    {
        msg = "El archivo ya existe";
        return -2;
    }
    std::ios::openmode modo_archivo = std::ios::out;
    if(modo == ModoDeEscritura::concatenar)
        modo_archivo |= std::ios::app;
    else
        modo_archivo |= std::ios::trunc;
    std::ofstream salida (archivo, modo_archivo);
    if(!salida.is_open())
    {
        return -1;
    }

    for(size_t i = 0; i< grupos.size(); i++)
    {
        salida<<grupos[i].Grupo()<<";"<<grupos[i].Residuo()<<std::endl;
    }
    salida.close();
    return 0;
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
