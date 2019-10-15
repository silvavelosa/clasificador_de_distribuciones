#include <string>
#include <sstream>
#include <UnitTest++.h>
#include <utility>
#include <vector>

#include "componentes_compartidos/entidades.h"

using namespace std::rel_ops;
using std::string;
using std::stringstream;
using std::vector;

namespace clasificador_de_distribuciones
{
namespace componentes_compartidos
{
namespace test
{
SUITE(ValidacionTest)
{
    TEST(Mayor)
    {
        Validacion a,b;
        a.id_ciudadano_ = 123456;
        a.score_ = 123;
        b.id_ciudadano_ = 12345;
        b.score_ = 125;
        CHECK_EQUAL(true, b<a);
        CHECK_EQUAL(false, a<b);

        b.id_ciudadano_ = 123456;
        CHECK_EQUAL(true, a<b);
        CHECK_EQUAL(false, b<a);

        b.score_ = 123;
        CHECK_EQUAL(false, a<b);
        CHECK_EQUAL(false, b<a);
    }

    TEST (Igual)
    {
        Validacion a,b;
        a.id_ciudadano_ = b.id_ciudadano_ = 123456;
        a.dia_ = b.dia_ = 1000;
        a.id_dedo_ = b.id_dedo_ = 7;
        a.score_ = b.score_ = 126;
        a.calidad_ = b.calidad_ = 157;
        CHECK(a==b);
        b.id_ciudadano_++;
        CHECK(a!=b);

    }
    TEST(Salida)
    {
        Validacion a;
        a.id_ciudadano_  = 123456;
        a.dia_  = 1000;
        a.id_dedo_  = 7;
        a.score_  = 126;
        a.calidad_  = 157;
        stringstream salida;
        salida<<a;
        CHECK_EQUAL("idCiudadano: 123456 dia: 1000 idDedo: 7 score: 126 calidad: 157",salida.str());

    }
    TEST(ParseExitoso)
    {
        string linea = "123456789;1200;2;147;124";
        Validacion res;
        int est = Validacion::Parse(linea,res);
        REQUIRE CHECK_EQUAL(Validacion::ParseResult::OK, est);
        CHECK (res.id_ciudadano_ == 123456789 &&
               res.dia_ == 1200 &&
               res.id_dedo_ == 2 &&
               res.score_ == 147 &&
               res.calidad_ == 124);
    }

    TEST(ParseSeparadorPersonalizadoExitoso)
    {
        string linea = "123456789,1200,2,147,124";
        Validacion res;
        int est = Validacion::Parse(linea, ',',res);
        REQUIRE CHECK_EQUAL(Validacion::ParseResult::OK, est);
        CHECK (res.id_ciudadano_ == 123456789 &&
               res.dia_ == 1200 &&
               res.id_dedo_ == 2 &&
               res.score_ == 147 &&
               res.calidad_ == 124);
    }

    TEST(ParseSaltoLinea)
    {
        string linea = "123456789;1200;2;147;124\n";
        Validacion res;
        int est = Validacion::Parse(linea,res);
        REQUIRE CHECK_EQUAL(Validacion::ParseResult::OK, est);
        CHECK (res.id_ciudadano_ == 123456789 &&
               res.dia_ == 1200 &&
               res.id_dedo_ == 2 &&
               res.score_ == 147 &&
               res.calidad_ == 124);
    }

    TEST(ParseEspacios)
    {
        string linea = "   123456789  ;  1200 ; 2;147 ;124 \n";
        Validacion res;
        int est = Validacion::Parse(linea,res);
        REQUIRE CHECK_EQUAL(Validacion::ParseResult::OK, est);
        CHECK (res.id_ciudadano_ == 123456789 &&
               res.dia_ == 1200 &&
               res.id_dedo_ == 2 &&
               res.score_ == 147 &&
               res.calidad_ == 124);
    }

    TEST(ParseSeparadorEspacioMultiple)
    {
        string linea = " 123456789   1200 2  147 124 \n";
        Validacion res;
        int est = Validacion::Parse(linea, ' ',res);
        REQUIRE CHECK_EQUAL(Validacion::ParseResult::OK, est);
        CHECK (res.id_ciudadano_ == 123456789 &&
               res.dia_ == 1200 &&
               res.id_dedo_ == 2 &&
               res.score_ == 147 &&
               res.calidad_ == 124);
    }

    TEST(ParseSeparadorEspacio)
    {
        string linea = "123456789 1200 2 147 124 ";
        Validacion res;
        int est = Validacion::Parse(linea, ' ',res);
        REQUIRE CHECK_EQUAL(Validacion::ParseResult::OK, est);
        CHECK (res.id_ciudadano_ == 123456789 &&
               res.dia_ == 1200 &&
               res.id_dedo_ == 2 &&
               res.score_ == 147 &&
               res.calidad_ == 124);
    }

    TEST(ParseDatosInvalidos)
    {
        string linea = "123456789;1da1;3;134;da";
        Validacion res;
        int est = Validacion::Parse(linea,res);
        CHECK_EQUAL(Validacion::ParseResult::CaracterInvalido, est);
    }

    TEST(ParseIdCiudadanoVacio)
    {
        string linea = ";1200;2;147;124";
        Validacion res;
        int est = Validacion::Parse(linea,res);
        CHECK_EQUAL(Validacion::ParseResult::IdCiudadanoVacio, est);
    }

    TEST(ParseIdCiudadanoEspacio)
    {
        string linea = " ;1200;2;147;124";
        Validacion res;
        int est = Validacion::Parse(linea,res);
        CHECK_EQUAL(Validacion::ParseResult::IdCiudadanoVacio, est);
    }

    TEST(ParseDiaVacio)
    {
        string linea = "123456789;;2;147;124";
        Validacion res;
        int est = Validacion::Parse(linea,res);
        CHECK_EQUAL(Validacion::ParseResult::DiaVacio, est);
    }

    TEST(ParseIdDedoVacio)
    {
        string linea = "123456789;1200;;147;124";
        Validacion res;
        int est = Validacion::Parse(linea,res);
        CHECK_EQUAL(Validacion::ParseResult::IdDedoVacio, est);
    }


    TEST(ParseScoreVacio)
    {
        string linea = "123456789;1200;2;;124";
        Validacion res;
        int est = Validacion::Parse(linea,res);
        CHECK_EQUAL(Validacion::ParseResult::ScoreVacio, est);
    }


    TEST(ParseCalidadVacio)
    {
        string linea = "123456789;1200;2;147;";
        Validacion res;
        int est = Validacion::Parse(linea,res);
        CHECK_EQUAL(Validacion::ParseResult::CalidadVacio, est);
    }

    TEST(ParseCalidadEspacio)
    {
        string linea = "123456789;1200;2;147; ";
        Validacion res;
        int est = Validacion::Parse(linea,res);
        CHECK_EQUAL(Validacion::ParseResult::CalidadVacio, est);
    }

    TEST(ParseCalidadSaltoLinea)
    {
        string linea = "123456789;1200;2;147; \n";
        Validacion res;
        int est = Validacion::Parse(linea,res);
        CHECK_EQUAL(Validacion::ParseResult::CalidadVacio, est);
    }

    TEST(ParseCadenaIncompleta)
    {
        string linea = "123456789;12";
        Validacion res;
        int est = Validacion::Parse(linea,res);
        CHECK_EQUAL(Validacion::ParseResult::LineaIncompleta, est);
    }

    TEST(ParseCadenaMuyLarga)
    {
        string linea = "123456789;1200;2;147;124;vnsvklala";
        Validacion res;
        int est = Validacion::Parse(linea,res);
        CHECK_EQUAL(Validacion::ParseResult::DatosSobrantes, est);
    }
}
SUITE (DistribucionTest)
{
    TEST (Diferencia)
    {
        Distribucion a,b;
        a.frecuencias_ = vector<int>{183, 69, 107, 127, 198, 110, 66, 148, 96, 84,
                                     61, 183, 112, 130, 160, 106, 84, 124, 7, 145,
                                     130, 165, 184, 190, 164, 31, 12, 71, 173, 130,
                                     22, 54, 168, 190, 195, 55, 83, 16, 66, 170};
        a.total_ = 4569;
        b.frecuencias_ = vector<int>{81, 122, 105, 67, 22, 24, 62, 192, 57, 48,
                                     9, 156, 182, 131, 118, 20, 30, 61, 158, 67,
                                     1, 34, 77, 53, 52, 191, 22, 72, 159, 197,
                                     172, 50, 6, 37, 38, 20, 146, 102, 83, 73};
        b.total_ = 3297;
        CHECK_CLOSE(0.0400525, a.FrecuenciaRelativa(0), 1e-6);

        CHECK_CLOSE(0.0005225, a.Diferencia(b), 1e-6);
    }
}
} // namespace test
} // namespace componentes_compartidos
} // namespace clasificador_de_distribuciones

int main (void)
{
    return UnitTest::RunAllTests();
}
