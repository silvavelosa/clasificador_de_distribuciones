#ifndef CLASIFICADOR_DE_DISTRIBUCIONES_COMPONENTES_COMPARTIDOS_ENTIDADES_H_
#define CLASIFICADOR_DE_DISTRIBUCIONES_COMPONENTES_COMPARTIDOS_ENTIDADES_H_

#include <iostream>
#include <string>
#include <vector>

namespace clasificador_de_distribuciones
{
namespace componentes_compartidos
{
class Evento
{
 public:
    enum ParseResult
    {
        OK = 0,
        IdGrupoVacio = -1,
        ValorVacio = -2,
        CaracterInvalido = -3,
        LineaIncompleta = -4,
        DatosSobrantes = -5
    };
    int id_grupo_;
    int valor_;
    static int Parse (const std::string& linea,
            Evento& evento);
    static int Parse (const std::string& linea,
            char separador,
            Evento& evento);
    const bool operator < (const Evento& a) const
    {
        return  id_grupo_<a.id_grupo_ ||
            (id_grupo_ == a.id_grupo_ && valor_ < a.valor_);
    }
    const bool operator == (const Evento& a) const
    {
        return  id_grupo_ == a.id_grupo_ && valor_ == a.valor_;
    }

    friend std::ostream &operator<<( std::ostream& output, const Evento &a ) {
     output << "idGrupo: "<< a.id_grupo_
            <<" valor: " << a.valor_;
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
