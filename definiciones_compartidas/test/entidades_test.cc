#include <string>
#include <UnitTest++.h>

#include "definiciones_compartidas/entidades.h"

using std::string;

namespace clasificador_de_distribuciones
{
namespace definiciones_compartidas
{
namespace test
{
SUITE(Validacion)
{
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
} // namespace test
} // namespace definiciones_compartidas
} // namespace clasificador_de_distribuciones

int main (void)
{
    return UnitTest::RunAllTests();
}
