#ifndef CLASIFICADOR_DE_DISTRIBUCIONES_DEFINICIONES_COMPARTIDAS_ENTIDADES_H_
#define CLASIFICADOR_DE_DISTRIBUCIONES_DEFINICIONES_COMPARTIDAS_ENTIDADES_H_

#include <string>
#include <vector>

namespace clasificador_de_distribuciones
{
namespace definiciones_compartidas
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
};

class Distribucion
{
 public:
    static const int tamano_frecuencias_;
    std::vector<int> frecuencias_;
    int total_;
    double diferencia_;
    Distribucion ()
    {
        if(tamano_frecuencias_ == 0)
            frecuencias_.resize(40);
        else
            frecuencias_.resize(tamano_frecuencias_);
        total_ = 0;
        diferencia_ = 0;
    }
    double FrecuenciaRelativa (int i) const
    {
        return (double)frecuencias_[i]/total_;
    }
    double Diferencia (const Distribucion& a) const;
};

} // namespace definiciones_compartidas
} // namespace clasificador_de_distribuciones

#endif // CLASIFICADOR_DE_DISTRIBUCIONES_DEFINICIONES_COMPARTIDAS_ENTIDADES_H_
