#include <memory>
#include <string>
#include <sstream>
#include <UnitTest++/UnitTest++.h>
#include <utility>
#include <vector>

#include "componentes_compartidos/entidades.h"

using namespace std::rel_ops;
using std::string;
using std::stringstream;
using std::unique_ptr;
using std::vector;

namespace clasificador_de_distribuciones
{
namespace componentes_compartidos
{
namespace test
{
SUITE(EventoTest)
{
    TEST(Mayor)
    {
        Evento a,b;
        a.id_grupo_ = 123456;
        a.valor_ = 123;
        b.id_grupo_ = 12345;
        b.valor_ = 125;
        CHECK_EQUAL(true, b<a);
        CHECK_EQUAL(false, a<b);

        b.id_grupo_ = 123456;
        CHECK_EQUAL(true, a<b);
        CHECK_EQUAL(false, b<a);

        b.valor_ = 123;
        CHECK_EQUAL(false, a<b);
        CHECK_EQUAL(false, b<a);
    }

    TEST (Igual)
    {
        Evento a,b;
        a.id_grupo_ = b.id_grupo_ = 123456;
        a.valor_ = b.valor_ = 157;
        CHECK(a==b);
        b.id_grupo_++;
        CHECK(a!=b);

    }
    TEST(Salida)
    {
        Evento a;
        a.id_grupo_  = 123456;
        a.valor_  = 1000;
        stringstream salida;
        salida<<a;
        CHECK_EQUAL("idGrupo: 123456 valor: 1000",salida.str());

    }
    TEST(ParseExitoso)
    {
        string linea = "123456789;1200";
        Evento res;
        int est = Evento::Parse(linea,res);
        REQUIRE CHECK_EQUAL(Evento::ParseResult::OK, est);
        CHECK (res.id_grupo_ == 123456789 &&
               res.valor_ == 1200);
    }

    TEST(ParseSeparadorPersonalizadoExitoso)
    {
        string linea = "123456789,1200";
        Evento res;
        int est = Evento::Parse(linea, ',',res);
        REQUIRE CHECK_EQUAL(Evento::ParseResult::OK, est);
        CHECK (res.id_grupo_ == 123456789 &&
               res.valor_ == 1200);
    }

    TEST(ParseSaltoLinea)
    {
        string linea = "123456789;1200\n";
        Evento res;
        int est = Evento::Parse(linea,res);
        REQUIRE CHECK_EQUAL(Evento::ParseResult::OK, est);
        CHECK (res.id_grupo_ == 123456789 &&
               res.valor_ == 1200);
    }

    TEST(ParseEspacios)
    {
        string linea = "   123456789  ;  1200  \n";
        Evento res;
        int est = Evento::Parse(linea,res);
        REQUIRE CHECK_EQUAL(Evento::ParseResult::OK, est);
        CHECK (res.id_grupo_ == 123456789 &&
               res.valor_ == 1200);
    }

    TEST(ParseSeparadorEspacioMultiple)
    {
        string linea = " 123456789   1200 \n";
        Evento res;
        int est = Evento::Parse(linea, ' ',res);
        REQUIRE CHECK_EQUAL(Evento::ParseResult::OK, est);
        CHECK (res.id_grupo_ == 123456789 &&
               res.valor_ == 1200);
    }

    TEST(ParseSeparadorEspacio)
    {
        string linea = "123456789 1200 ";
        Evento res;
        int est = Evento::Parse(linea, ' ',res);
        REQUIRE CHECK_EQUAL(Evento::ParseResult::OK, est);
        CHECK (res.id_grupo_ == 123456789 &&
               res.valor_ == 1200);
    }

    TEST(ParseDatosInvalidos)
    {
        string linea = "123456789;1da1";
        Evento res;
        int est = Evento::Parse(linea,res);
        CHECK_EQUAL(Evento::ParseResult::CaracterInvalido, est);
    }

    TEST(ParseIdGrupoVacio)
    {
        string linea = ";1200";
        Evento res;
        int est = Evento::Parse(linea,res);
        CHECK_EQUAL(Evento::ParseResult::IdGrupoVacio, est);
    }

    TEST(ParseIdGrupoEspacio)
    {
        string linea = " ;1200";
        Evento res;
        int est = Evento::Parse(linea,res);
        CHECK_EQUAL(Evento::ParseResult::IdGrupoVacio, est);
    }

    TEST(ParseValorVacio)
    {
        string linea = "123456789;";
        Evento res;
        int est = Evento::Parse(linea,res);
        CHECK_EQUAL(Evento::ParseResult::ValorVacio, est);
    }

    TEST(ParseValorEspacio)
    {
        string linea = "123456789; ";
        Evento res;
        int est = Evento::Parse(linea,res);
        CHECK_EQUAL(Evento::ParseResult::ValorVacio, est);
    }

    TEST(ParseValorSaltoLinea)
    {
        string linea = "123456789; \n";
        Evento res;
        int est = Evento::Parse(linea,res);
        CHECK_EQUAL(Evento::ParseResult::ValorVacio, est);
    }

    TEST(ParseCadenaIncompleta)
    {
        string linea = "";
        Evento res;
        int est = Evento::Parse(linea,res);
        CHECK_EQUAL(Evento::ParseResult::LineaIncompleta, est);
    }

    TEST(ParseCadenaMuyLarga)
    {
        string linea = "123456789;1200;2;147;124;vnsvklala";
        Evento res;
        int est = Evento::Parse(linea,res);
        CHECK_EQUAL(Evento::ParseResult::DatosSobrantes, est);
    }
}
SUITE (DistribucionTest)
{
    TEST (Diferencia)
    {
        Distribucion a(123,unique_ptr<vector<unsigned int> >(
                new vector<unsigned int> ({183, 69, 107, 127, 198, 110, 66, 148, 96, 84,
                                 61, 183, 112, 130, 160, 106, 84, 124, 7, 145,
                                 130, 165, 184, 190, 164, 31, 12, 71, 173, 130,
                                 22, 54, 168, 190, 195, 55, 83, 16, 66, 170}) ) );

        Distribucion b(127,unique_ptr<vector<unsigned int> >(
                new vector<unsigned int> ({81, 122, 105, 67, 22, 24, 62, 192, 57, 48,
                                 9, 156, 182, 131, 118, 20, 30, 61, 158, 67,
                                 1, 34, 77, 53, 52, 191, 22, 72, 159, 197,
                                 172, 50, 6, 37, 38, 20, 146, 102, 83, 73}) ) );

        CHECK_CLOSE(0.0400525, a.FrecuenciaRelativa(0), 1e-6);

        CHECK_CLOSE(0.0005225, a.Diferencia(b), 1e-6);
    }
}
} // namespace test
} // namespace componentes_compartidos
} // namespace clasificador_de_distribuciones
