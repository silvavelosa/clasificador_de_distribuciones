#include "componentes_compartidos/entidades.h"

#include <algorithm>
#include <cmath>

using namespace std;

namespace clasificador_de_distribuciones
{
namespace componentes_compartidos
{
const int Distribucion::tamano_frecuencias_ = 40;

double Distribucion::Diferencia (const Distribucion& a) const {
    double diff = 0;
    for(unsigned int i=0;
            i+1 < min(frecuencias_.size(), a.frecuencias_.size());
            i++)
    {
        diff += pow(FrecuenciaRelativa(i) - a.FrecuenciaRelativa(i), 2);
    }
    double miResto=0, aResto=0;

    for(unsigned int i=min(frecuencias_.size(), a.frecuencias_.size())-1;
            i < max(frecuencias_.size(), a.frecuencias_.size());
            i++)
    {
        if(frecuencias_.size() > i)
            miResto += FrecuenciaRelativa(i);
        if(a.frecuencias_.size() > i)
            aResto += a.FrecuenciaRelativa(i);
    }
    diff += pow(miResto - aResto, 2);
    diff/= min(frecuencias_.size(), a.frecuencias_.size());
    return diff;
}

int Evento::Parse (const string& linea,
        char separador,
        Evento& evento)
{
    unsigned int i=0;
    int seccion = 0;
    for(seccion = 0; seccion < 2 && i < linea.size(); seccion++)
    {

        switch(seccion)
        {
        case 0:
            evento.id_grupo_ = atoi(linea.c_str() + i);
            break;
        case 1:
            evento.valor_ = atoi(linea.c_str() + i);
            break;
        }
        for(; i<linea.size() && linea[i] == ' '; i++);

        if(i == linea.size() || linea[i] == separador || linea[i] == '\n')
            return -(seccion+1);

        for(; i<linea.size() && linea[i]!= separador && linea[i] != '\n'; i++)
        {
            if((linea[i] < '0' || linea[i] > '9') && linea[i] != ' ')
                return Evento::ParseResult::CaracterInvalido;
        }

        if(linea[i]=='\n')
            i=linea.size();

        if(i<linea.size())
            i++;
    }
    if(seccion == 1)
        return Evento::ParseResult::ValorVacio;
    if(seccion < 1)
        return Evento::ParseResult::LineaIncompleta;

    if(i<linea.size() && separador == ' ')
    {
        for(; i<linea.size() && linea[i] == ' '; i++);

        if(linea[i]=='\n')
            i=linea.size();
    }
    if(i<linea.size())
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

