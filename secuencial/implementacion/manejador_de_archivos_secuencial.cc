#include "secuencial/implementacion/manejador_de_archivos_secuencial.h"

#include <fstream>
#include <iostream>
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

int ManejadorDeArchivosSecuencial::CargarDatos(const string& archivo,
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

int ManejadorDeArchivosSecuencial::GenerarSalida(const string& archivo,
        const map<int,Distribucion>& ciudadanos,
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

    map<int,Distribucion>::const_iterator it;
    for(it = ciudadanos.cbegin(); it!= ciudadanos.cend(); it++)
    {
        salida<<it->first<<";"<<it->second.diferencia_<<std::endl;
    }
    salida.close();
    return 0;
}

int ManejadorDeArchivosSecuencial::GenerarSalida(const string& archivo,
        const vector<map<int,Distribucion>::const_iterator>& indice,
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

    for(unsigned int i = 0; i< indice.size(); i++)
    {
        salida<<indice[i]->first<<";"<<indice[i]->second.diferencia_<<std::endl;
    }
    salida.close();
    return 0;
}

int ManejadorDeArchivosSecuencial::TamanoDeArchivo(const string& archivo)
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
} // namespace secuencial
} // namespace clasificador_de_distribuciones
