#include <ctime>
#include <iostream>
#include <memory>
#include <vector>

#include <omp.h>

#include "componentes_compartidos/entidades.h"
#include "open_mp/implementacion/manejador_de_archivos_open_mp.h"
#include "open_mp/implementacion/analizador_de_datos_open_mp.h"

using namespace clasificador_de_distribuciones::componentes_compartidos;
using namespace clasificador_de_distribuciones::open_mp::implementacion;
using namespace std;


int main (int argc, char** argv) {

    clock_t inicioT = clock();
    clock_t inicio = inicioT;
    Distribucion::EstablecerTamanoFrecuencias(24);
    Distribucion::EstablecerTamanoIntervalos(1);
    if (argc < 3 || argc > 4) {
        cout<<"uso: ./clasificador_de_distribuciones_open_mp "
            <<"archivo_entrada "
            <<"archivo_salida "
            <<"[hilos]"<<endl;
        return 0;
    }

    int stat= 0;
    string msg;
    unique_ptr<vector<Evento> > eventos;
    string archivo_entrada = argv[1];
    string archivo_salida = argv[2];
    unsigned int nth = omp_get_num_procs();

    if(argc == 4)
    {
        nth = atoi(argv[3]);
    }
    ManejadorDeArchivosOpenMP manejador_de_archivos(nth);
    stat = manejador_de_archivos.CargarDatos( archivo_entrada, eventos, msg);
    /*  +++
        Varios hilos pueden leer simultaneamente el archivo
        para 10 millones de registros, el dataset pesa al rededor de 300 MB
    */
    switch (stat)
    {
    case 0:
        cout<<"Lectura finalizada - total eventos: "<<eventos->size()
            <<" duracion: "<<(double)(clock()-inicio)/CLOCKS_PER_SEC<<endl;

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
    inicio = clock();
    AnalizadorDeDatosOpenMP analizador_de_datos(nth);
    stat = analizador_de_datos.OrdenarEventos(eventos);
    /*  +++
        El ordenamiento se puede paralelizar con el algoritmo Merge Sort
    */

    switch (stat)
    {
    case 0:
        cout<<"Ordenamiento de eventos finalizado"
            <<" duracion: "<<(double)(clock()-inicio)/CLOCKS_PER_SEC<<endl;
        break;
    case -1:
        cout<<"Error al ordenar:"<<endl;
        return 0;
        break;
    }

    unique_ptr<Distribucion> promedio;
    unique_ptr<vector<Distribucion> > grupos;

    stat = analizador_de_datos.AgruparYPromediar(*eventos,
                                            grupos,
                                            promedio);

    /*  +++
        Paralelizacion por datos, cada hilo puede procesar un lote de
        eventos, como las eventos de un grupo están cerca
        en el arreglo de eventos, los conflictos por varios hilos
        intentado actualizar el mismo grupo se verán minimizados.
    */

    switch (stat)
    {
    case 0:
        cout<<"Agrupamiento y promedio finalizados - total grupos: "<<
            grupos->size()<<endl;

        break;
    case -1:
        cout<<"Error al agrupar y promediar"<<endl;
        return 0;
        break;
    }

    stat = analizador_de_datos.CompararDistribuciones(*grupos, *promedio);
    /*  +++
        Paralelizacion por datos, cada hilo puede procesar un lote de
        grupos, la única memoria que necesitan todos los hilos
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

    stat = analizador_de_datos.RegresionLineal(*grupos);

    /* +/-
        probablemente sea paralelizable, sin embargo, hace falta revisar a detalle
        el codigo de la libreria que calcula la regresion para evaluarlo.
    */
    switch (stat)
    {
    case 0:
        cout<<"Regresion lineal y calculo de residuos finalizada"<<endl;
        break;
    case -1:
        cout<<"Error al calcular residuos"<<endl;
        return 0;
        break;
    }

    stat = analizador_de_datos.OrdenarDistribuciones(grupos);
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
                                               *grupos,
                                               msg,
                                               ManejadorDeArchivosOpenMP::reemplazar);
    /*  ---
        No paralelizable - sólo un hilo puede escribir a la vez
    */
    switch (stat)
    {
    case 0:
        cout<<"Generación de archivo de salida finalizada"
        <<" duracion TOTAL: "<<(double)(clock()-inicioT)/CLOCKS_PER_SEC<<endl;
        break;
    case -1:
        cout<<"Error al generar archivo de salida:"<<endl<<msg<<endl;
        return 0;
        break;
    }

    return 0;
}
