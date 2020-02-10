#ifndef CLASIFICADOR_DE_DISTRIBUCIONES_COMPONENTES_COMPARTIDOS_ENTIDADES_H_
#define CLASIFICADOR_DE_DISTRIBUCIONES_COMPONENTES_COMPARTIDOS_ENTIDADES_H_

#ifndef __CUDACC__
#define __device__
#define __host__
#endif

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
    static int Parse (const char* linea,
            char separador,
            Evento& evento,
            size_t* avance = nullptr);
    __device__ __host__ Evento& operator= ( const Evento& a)
    {
        id_grupo_ = a.id_grupo_;
        valor_ = a.valor_;
        return *this;
    }
    __device__ __host__ bool operator < (const Evento& a) const
    {
        return  id_grupo_<a.id_grupo_ ||
            (id_grupo_ == a.id_grupo_ && valor_ < a.valor_);
    }
    __device__ __host__ bool operator == (const Evento& a) const
    {
        return  id_grupo_ == a.id_grupo_ && valor_ == a.valor_;
    }

    friend std::ostream& operator <<( std::ostream& output, const Evento &a );
};

class Distribucion
{
 public:
    static void EstablecerTamanoFrecuencias (const size_t& tamano_frecuencias)
    {
        if(tamanos_actualizables_)
            tamano_frecuencias_ = tamano_frecuencias;
    }
    static void EstablecerTamanoIntervalos (const size_t& tamano_intervalos)
    {
        if(tamanos_actualizables_)
            tamano_intervalos_ = tamano_intervalos;
    }
    static const size_t& TamanoFrecuencias()
    {
        return tamano_frecuencias_;
    }
    static const size_t& TamanoIntervalos()
    {
        return tamano_intervalos_;
    }
    Distribucion (int id_grupo)
    {
        tamanos_actualizables_ = false;
        id_grupo_ = id_grupo;
        frecuencias_.reset(new std::vector<unsigned int>(tamano_frecuencias_));
        total_ = 0;
        diferencia_ = 0;
        residuo_ = 0;
    }
    Distribucion(): Distribucion(0){};
    Distribucion(int id_grupo, std::unique_ptr<std::vector<unsigned int> > frecuencias)
            : Distribucion(id_grupo)
    {
        frecuencias_ = std::move(frecuencias);
        for(size_t i=tamano_frecuencias_;i < frecuencias_->size();i++)
        {
                (*frecuencias_)[tamano_frecuencias_-1]+=(*frecuencias_)[i];
        }
        frecuencias_->resize(tamano_frecuencias_);
        for(size_t i = 0; i < tamano_frecuencias_; i++)
            total_ += (*frecuencias_)[i];

    }
    void operator += (const Distribucion& b) 
    {
        for(size_t i = 0; i < tamano_frecuencias_; i++)
            (*frecuencias_)[i]+=(*b.frecuencias_)[i];
        total_ += b.total_;
    }

    double FrecuenciaRelativa (int i) const
    {
        return (double)(*frecuencias_)[i]/total_;
    }
    double Diferencia (const Distribucion& a) const;
    void AnadirEvento(const Evento& evento);
    const std::vector<unsigned int>& Frecuencias()
    {
        return *frecuencias_;
    }
    int Grupo() const
    {
        return id_grupo_;
    }
    unsigned int Total() const
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
    void EstablecerDiferencia(const double& diferencia)
    {
        diferencia_ = diferencia;
    }
    void EstablecerDiferencia(const Distribucion& a)
    {
        diferencia_ = Diferencia(a);
    }
    void EstablecerResiduo(const double& residuo)
    {
        residuo_ = residuo;
    }
 private:
    static size_t tamano_frecuencias_;
    static size_t tamano_intervalos_;
    static bool tamanos_actualizables_;
    std::unique_ptr<std::vector<unsigned int> > frecuencias_;
    int id_grupo_;
    unsigned int total_;
    double diferencia_;
    double residuo_;
};

} // namespace componentes_compartidos
} // namespace clasificador_de_distribuciones

#endif // CLASIFICADOR_DE_DISTRIBUCIONES_COMPONENTES_COMPARTIDOS_ENTIDADES_H_
