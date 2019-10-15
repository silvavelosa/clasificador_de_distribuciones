#ifndef CLASIFICADOR_DE_DISTRIBUCIONES_COMPONENTES_COMPARTIDOS_ENTIDADES_H_
#define CLASIFICADOR_DE_DISTRIBUCIONES_COMPONENTES_COMPARTIDOS_ENTIDADES_H_

#include <iostream>
#include <string>
#include <vector>

namespace clasificador_de_distribuciones
{
namespace componentes_compartidos
{
class Validacion
{
 public:
    enum ParseResult
    {
        OK = 0,
        IdCiudadanoVacio = -1,
        DiaVacio = -2,
        IdDedoVacio = -3,
        ScoreVacio = -4,
        CalidadVacio = -5,
        CaracterInvalido = -6,
        LineaIncompleta = -7,
        DatosSobrantes = -8
    };
    int id_ciudadano_;
    int dia_;
    int id_dedo_;
    int score_;
    int calidad_;
    static int Parse (const std::string& linea,
            Validacion& validacion);
    static int Parse (const std::string& linea,
            char separador,
            Validacion& validacion);
    const bool operator < (const Validacion& a) const
    {
        return  id_ciudadano_<a.id_ciudadano_ ||
            (id_ciudadano_ == a.id_ciudadano_ && score_ < a.score_);
    }
    const bool operator == (const Validacion& a) const
    {
        return  id_ciudadano_== a.id_ciudadano_ &&
            dia_             == a.dia_          &&
            id_dedo_         == a.id_dedo_      &&
            score_           == a.score_        &&
            calidad_         == a.calidad_;
    }

    friend std::ostream &operator<<( std::ostream& output, const Validacion &a ) {
     output << "idCiudadano: "<< a.id_ciudadano_
            <<" dia: " << a.dia_
            << " idDedo: "<<a.id_dedo_
            << " score: "<<a.score_
            << " calidad: "<<a.calidad_;
     return output;
    }
};

class Distribucion
{
 public:
    static const int tamano_frecuencias_;
    std::vector<int> frecuencias_;
    int total_;
    double diferencia_;
    double residuo_;
    Distribucion ()
    {
        frecuencias_.resize(tamano_frecuencias_);
        total_ = 0;
        diferencia_ = 0;
        residuo_ = 0;
    }
    double FrecuenciaRelativa (int i) const
    {
        return (double)frecuencias_[i]/total_;
    }
    double Diferencia (const Distribucion& a) const;
};

} // namespace componentes_compartidos
} // namespace clasificador_de_distribuciones

#endif // CLASIFICADOR_DE_DISTRIBUCIONES_COMPONENTES_COMPARTIDOS_ENTIDADES_H_
