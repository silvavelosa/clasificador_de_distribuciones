#include <iostream>
#include <memory>
#include <vector>

#include "componentes_compartidos/entidades.h"
#include "secuencial/implementacion/manejador_de_archivos_secuencial.h"
#include "secuencial/implementacion/analizador_de_datos_secuencial.h"

using namespace clasificador_de_distribuciones::componentes_compartidos;
using namespace clasificador_de_distribuciones::secuencial::implementacion;
using namespace std;


int main (int argc, char** argv) {
    if (argc != 3) {
        printf("uso: ./clasificador_de_distribuciones_secuencial archivo_entrada archivo_salida");
        return 0;
    }

    int stat= 0;
    string msg;
    unique_ptr<vector<Validacion> > validaciones;
    string archivo_entrada = argv[1];
    string archivo_salida = argv[2];

    ManejadorDeArchivosSecuencial manejador_de_archivos;
    stat = manejador_de_archivos.CargarDatos( archivo_entrada, validaciones, msg);
    /*  +++
        Varios hilos pueden leer simultaneamente el archivo
        para 10 millones de registros, el dataset pesa al rededor de 300 MB
    */
    switch (stat)
    {
    case 0:
        cout<<"Lectura finalizada - total validaciones: "<<validaciones->size()<<endl;

        break;
    case -1:
        cout<<"El archivo "<<archivo_entrada<<" no se encontró"<<endl;
        return 0;
        break;
    case -2:
        cout<<"Error al procesar el archivo:"<<endl<<msg<<endl;
        return 0;
        break;
    }

    AnalizadorDeDatosSecuencial analizador_de_datos;
    stat = analizador_de_datos.OrdenarValidaciones(validaciones);
    /*  +++
        El ordenamiento se puede paralelizar con el algoritmo Merge Sort
    */

    switch (stat)
    {
    case 0:
        cout<<"Ordenamiento de validaciones finalizado"<<endl;
        break;
    case -1:
        cout<<"Error al ordenar:"<<endl;
        return 0;
        break;
    }

    unique_ptr<Distribucion> promedio;
    unique_ptr<map<int,Distribucion> > ciudadanos;

    stat = analizador_de_datos.AgruparYPromediar(*validaciones,
                                            ciudadanos,
                                            promedio);

    /*  +++
        Paralelizacion por datos, cada hilo puede procesar un lote de
        validaciones, como las validaciones de un ciudadano están cerca
        en el arreglo de validaciones, los conflictos por varios hilos
        intentado actualizar el mismo ciudadano se verán minimizados.
    */

    switch (stat)
    {
    case 0:
        cout<<"Agrupamiento y promedio finalizados - total ciudadanos: "<<
            ciudadanos->size()<<endl;

        break;
    case -1:
        cout<<"Error al agrupar y promediar"<<endl;
        return 0;
        break;
    }

    stat = analizador_de_datos.CompararDistribuciones(ciudadanos, *promedio);
    /*  +++
        Paralelizacion por datos, cada hilo puede procesar un lote de
        ciudadanos, la única memoria que necesitan todos los hilos
        es la distribución promedio, sin embargo esta sólo es leída,
        así que no debería haber conflictos.
    */
    switch (stat)
    {
    case 0:
        cout<<"Comparacion de distribuciones finalizada"<<endl;
        break;
    case -1:
        cout<<"Error al comparar distribuciones"<<endl;
        return 0;
        break;
    }

    unique_ptr<vector<map<int,Distribucion>::const_iterator> > indice;

    stat = analizador_de_datos.OrdenarDistribuciones(*ciudadanos, indice);
    /*  +++
        Paralelizable - ordenamiento.
    */
    switch (stat)
    {
    case 0:
        cout<<"Ordenamiento de distribuciones finalizada"<<endl;
        break;
    case -1:
        cout<<"Error al ordenar distribuciones"<<endl;
        return 0;
        break;
    }

    stat = manejador_de_archivos.GenerarSalida(archivo_salida,
                                               *indice,
                                               msg);
    /*  ---
        No paralelizable - sólo un hilo puede escribir a la vez
    */
    switch (stat)
    {
    case 0:
        cout<<"Generación de archivo de salida finalizada"<<endl;
        break;
    case -1:
        cout<<"Error al generar archivo de salida:"<<endl<<msg<<endl;
        return 0;
        break;
    }

    return 0;
}
