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
    int tamano = TamanoDeArchivo(archivo);

    eventos.reset(new vector<Evento>);
    eventos->resize(tamano/5);

    std::ifstream entrada(archivo);
    if(!entrada.is_open() || tamano <= 0)
    {
        eventos.reset(nullptr);
        return -1;
    }

    char* contenido = (char*) malloc(tamano*sizeof(char) + 100);
    entrada.read(contenido, tamano + 100);
    size_t n_caracteres = entrada.gcount();

    for(size_t i = n_caracteres; i< tamano + 100; i++)
        contenido[i] = '\0';

    size_t n_eventos=0,pos = 0;
    while(pos < n_caracteres)
    {
        size_t avance;
        int est = Evento::Parse(contenido+pos,';', (*eventos)[n_eventos++], &avance);
        pos += avance+1;
        if(est != Evento::ParseResult::OK)
        {
            std::stringstream ss;
            switch(est)
            {
            case Evento::ParseResult::IdGrupoVacio:
                ss<<"IdGrupo vacio en la linea "<<n_eventos;
                break;
            case Evento::ParseResult::ValorVacio:
                ss<<"Valor vacio en la linea "<<n_eventos;
                break;
            case Evento::ParseResult::CaracterInvalido:
                ss<<"Caracter invalido en la linea "<<n_eventos;
                break;
            case Evento::ParseResult::LineaIncompleta:
                ss<<"Linea incompleta "<<n_eventos;
                break;
            case Evento::ParseResult::DatosSobrantes:
                ss<<"Datos sobrantes en la linea "<<n_eventos;
                break;
            }
            eventos.reset(nullptr);
            msg = ss.str();
            free(contenido);
            return -2;
        }
    }
    entrada.close();
    eventos->resize(n_eventos);
    eventos->shrink_to_fit();
    free(contenido);
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
