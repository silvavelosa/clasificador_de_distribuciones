#include "componentes_compartidos/entidades.h"

#include <algorithm>
#include <cmath>

using namespace std;

namespace clasificador_de_distribuciones
{
namespace componentes_compartidos
{

size_t Distribucion::tamano_frecuencias_ = 40;
size_t Distribucion::tamano_intervalos_ = 10;
bool Distribucion::tamanos_actualizables_ = true;

void Distribucion::AnadirEvento (const Evento& evento)
{
    unsigned int intervalo = ((unsigned int)evento.valor_)/tamano_intervalos_;
    if(intervalo >= tamano_frecuencias_)
    {
        #pragma omp atomic
        (*frecuencias_)[tamano_frecuencias_-1]++;
    }
    else
    {
        #pragma omp atomic
        (*frecuencias_)[intervalo]++;
    }

    #pragma omp atomic
    total_++;
}

double Distribucion::Diferencia (const Distribucion& a) const {
    double diff = 0;
    for(size_t i=0; i < Distribucion::tamano_frecuencias_; i++)
    {
        diff += pow(FrecuenciaRelativa(i) - a.FrecuenciaRelativa(i), 2);
    }
    diff/= Distribucion::tamano_frecuencias_;
    return diff;
}

int Evento::Parse (const string& linea,
        char separador,
        Evento& evento)
{
    return Parse(linea.data(),separador, evento);
}

int Evento::Parse (const char* linea,
        char separador,
        Evento& evento)
{
    size_t i=0;
    int seccion = 0;
    for(seccion = 0; seccion < 2 && linea[i]!= '\0' && linea[i] != '\n'; seccion++)
    {

        switch(seccion)
        {
        case 0:
            evento.id_grupo_ = atoi(linea + i);
            break;
        case 1:
            evento.valor_ = atoi(linea + i);
            break;
        }
        for(; linea[i]!='\0' && linea[i]!='\n' && linea[i] == ' '; i++);

        if(linea[i] == '\0' || linea[i] == separador || linea[i] == '\n')
            return -(seccion+1);

        for(; linea[i] != '\0' && linea[i]!= separador && linea[i] != '\n'; i++)
        {
            if((linea[i] < '0' || linea[i] > '9') && linea[i] != ' ')
                return Evento::ParseResult::CaracterInvalido;
        }

        if(linea[i]!='\n' && linea[i] != '\0')
            i++;
    }
    if(seccion == 1)
        return Evento::ParseResult::ValorVacio;
    if(seccion < 1)
        return Evento::ParseResult::LineaIncompleta;

    if(linea[i]!='\0' && separador == ' ')
    {
        for(;linea[i] == ' '; i++);
    }
    if(linea[i]!='\0' && linea[i] != '\n')
    {
        return Evento::ParseResult::DatosSobrantes;
    }
    return Evento::ParseResult::OK;
}

int Evento::Parse (const string& linea, Evento& evento) {
    return Parse(linea, ';', evento);
}
} // namespace componentes_compartidos
} // namespace clasificador_de_distribuciones

