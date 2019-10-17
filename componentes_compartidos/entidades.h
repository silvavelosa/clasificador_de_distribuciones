#ifndef CLASIFICADOR_DE_DISTRIBUCIONES_COMPONENTES_COMPARTIDOS_ENTIDADES_H_
#define CLASIFICADOR_DE_DISTRIBUCIONES_COMPONENTES_COMPARTIDOS_ENTIDADES_H_

#include <iostream>
#include <memory>
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
    static const unsigned int tamano_frecuencias_;
    static const unsigned int tamano_intervalos_;
    Distribucion (int id_grupo)
    {
        id_grupo_ = id_grupo;
        frecuencias_.reset(new std::vector<int>(tamano_frecuencias_));
        total_ = 0;
        diferencia_ = 0;
        residuo_ = 0;
    }
    Distribucion(): Distribucion(0){};
    Distribucion(int id_grupo, std::unique_ptr<std::vector<int> > frecuencias)
            : Distribucion(id_grupo)
    {
        frecuencias_ = std::move(frecuencias);
        for(unsigned int i=tamano_frecuencias_;i<frecuencias_->size();i++)
        {
                (*frecuencias_)[tamano_frecuencias_-1]+=(*frecuencias_)[i];
        }
        frecuencias_->resize(tamano_frecuencias_);
        for(unsigned int i=0;i<tamano_frecuencias_;i++)
            total_ += (*frecuencias_)[i];

    }

    double FrecuenciaRelativa (int i) const
    {
        return (double)(*frecuencias_)[i]/total_;
    }
    double Diferencia (const Distribucion& a) const;
    void AnadirEvento(const Evento& evento)
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
    const std::vector<int>& Frecuencias()
    {
        return *frecuencias_;
    }
    int Grupo() const
    {
        return id_grupo_;
    }
    int Total() const
    {
        return total_;
    }
    double Diferencia()const
    {
        return diferencia_;
    }
    double Residuo() const
    {
        return residuo_;
    }
    void EstablecerDiferencia(double diferencia)
    {
        diferencia_ = diferencia;
    }
    void EstablecerDiferencia(const Distribucion& a)
    {
        diferencia_ = Diferencia(a);
    }
    void EstablecerResiduo(double residuo)
    {
        residuo_ = residuo;
    }
 private:
    std::unique_ptr<std::vector<int> > frecuencias_;
    int id_grupo_;
    int total_;
    double diferencia_;
    double residuo_;
};

} // namespace componentes_compartidos
} // namespace clasificador_de_distribuciones

#endif // CLASIFICADOR_DE_DISTRIBUCIONES_COMPONENTES_COMPARTIDOS_ENTIDADES_H_
