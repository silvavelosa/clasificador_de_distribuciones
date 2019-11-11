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
    char linea[25];
    int tamano = TamanoDeArchivo(archivo);

    eventos.reset(new vector<Evento>);
    eventos->resize(tamano/5);

    std::ifstream entrada(archivo);
    if(!entrada.is_open() || tamano <= 0)
    {
        eventos.reset(nullptr);
        return -1;
    }

    size_t i;
    for(i=1;entrada.getline(linea, 25);i++)
    {
        int est = Evento::Parse(linea, (*eventos)[i-1]);

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
                ss<<"Datos sobrantes en la linea "<<i;
                break;
            }
            eventos.reset(nullptr);
            msg = ss.str();
            return -2;
        }
    }
    entrada.close();
    eventos->resize(i-1);
    eventos->shrink_to_fit();
    return 0;
}

int ManejadorDeArchivosSecuencial::GenerarSalida(const string& archivo,
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
