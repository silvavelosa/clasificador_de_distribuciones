#include "componentes_compartidos/entidades.h"

#include <algorithm>
#include <cmath>

using namespace std;

namespace clasificador_de_distribuciones
{
namespace componentes_compartidos
{
double Distribucion::Diferencia (const Distribucion& a) const {
    double diff = 0;
    for(int i=0;
            i+1 < min(frecuencias_.size(), a.frecuencias_.size());
            i++)
    {
        diff += pow(FrecuenciaRelativa(i) - a.FrecuenciaRelativa(i), 2);
    }
    double miResto=0, aResto=0;

    for(int i=min(frecuencias_.size(), a.frecuencias_.size())-1;
            i < max(frecuencias_.size(), a.frecuencias_.size());
            i++)
    {
        if(frecuencias_.size() > i)
            miResto += FrecuenciaRelativa(i);
        if(a.frecuencias_.size() > i)
            aResto += a.FrecuenciaRelativa(i);
    }
    diff += pow(miResto - aResto, 2);
    return diff;
}

int Validacion::Parse (const string& linea,
        char separador,
        Validacion& validacion)
{
    int i=0;
    int seccion = 0;
    for(seccion = 0; seccion < 5 && i < linea.size(); seccion++)
    {

        switch(seccion)
        {
        case 0:
            validacion.id_ciudadano_ = atoi(linea.c_str() + i);
            break;
        case 1:
            validacion.dia_ = atoi(linea.c_str() + i);
            break;
        case 2:
            validacion.id_dedo_ = atoi(linea.c_str() + i);
            break;
        case 3:
            validacion.score_ = atoi(linea.c_str() + i);
            break;
        case 4:
            validacion.calidad_ = atoi(linea.c_str() + i);
            break;
        }
        for(; i<linea.size() && linea[i] == ' '; i++);

        if(i == linea.size() || linea[i] == separador || linea[i] == '\n')
            return -(seccion+1);

        for(; i<linea.size() && linea[i]!= separador && linea[i] != '\n'; i++)
        {
            if((linea[i] < '0' || linea[i] > '9') && linea[i] != ' ')
                return Validacion::ParseResult::CaracterInvalido;
        }

        if(linea[i]=='\n')
            i=linea.size();

        if(i<linea.size())
            i++;
    }
    if(seccion == 4)
        return Validacion::ParseResult::CalidadVacio;
    if(seccion < 4)
        return Validacion::ParseResult::LineaIncompleta;

    if(i<linea.size() && separador == ' ')
    {
        for(; i<linea.size() && linea[i] == ' '; i++);

        if(linea[i]=='\n')
            i=linea.size();
    }
    if(i<linea.size())
    {
        return Validacion::ParseResult::DatosSobrantes;
    }
    return Validacion::ParseResult::OK;
}

int Validacion::Parse (const string& linea, Validacion& validacion) {
    return Parse(linea, ';', validacion);
}
} // namespace componentes_compartidos
} // namespace clasificador_de_distribuciones

