#ifndef CLASIFICADOR_DE_DISTRIBUCIONES_COMPONENTES_COMPARTIDOS_UTIL_ARCHIVOS_H_
#define CLASIFICADOR_DE_DISTRIBUCIONES_COMPONENTES_COMPARTIDOS_UTIL_ARCHIVOS_H_
#include <string>
#include <fstream>
#include <iostream>

namespace clasificador_de_distribuciones
{
namespace componentes_compartidos
{
int TamanoDeArchivo(const std::string& archivo)
{

    std::ifstream entrada (archivo, std::ios::binary);
    if(!entrada.is_open()) 
        return -1;
    std::streampos ini = entrada.tellg();
    entrada.seekg (0, std::ios::end);
    std::streampos fin = entrada.tellg();
    entrada.close();
    return fin-ini;
}
} // namespace componentes_compartidos
} // namespace clasificador_de_distribuciones
#endif //CLASIFICADOR_DE_DISTRIBUCIONES_COMPONENTES_COMPARTIDOS_UTIL_ARCHIVOS_H_