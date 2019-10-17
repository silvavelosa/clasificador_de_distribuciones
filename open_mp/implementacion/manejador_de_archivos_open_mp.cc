#include "open_mp/implementacion/manejador_de_archivos_open_mp.h"

#include <fstream>
#include <iostream>
#include <sstream>

#include <omp.h>

namespace clasificador_de_distribuciones
{
namespace open_mp
{
namespace implementacion
{
using std::map;
using std::string;
using std::unique_ptr;
using std::vector;

int ManejadorDeArchivosOpenMP::CargarDatos(const string& archivo,
        unique_ptr<vector<Evento> >& eventos,
        string& msg) {
    string linea;

    int tamano = TamanoDeArchivo(archivo);

    Evento actual;
    eventos.reset(new vector<Evento>);
    eventos->reserve(tamano/5);


    std::ifstream entrada(archivo);
    if(!entrada.is_open() || tamano <= 0)
    {
        eventos.reset(nullptr);
        return -1;
    }

    for(int i=1;getline(entrada, linea);i++)
    {
        int est = Evento::Parse(linea, actual);
        if(est == Evento::ParseResult::OK)
            eventos->push_back(actual);
        else
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
                ss<<"Datos sobrantes en la linea "<<i;
                break;
            }
            eventos.reset(nullptr);
            msg = ss.str();
            return -2;
        }
    }
    entrada.close();
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

    for(unsigned int i = 0; i< grupos.size(); i++)
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
