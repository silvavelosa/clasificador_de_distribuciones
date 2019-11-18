#include <ctime>
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
        cout<<"uso: ./clasificador_de_distribuciones_secuencial archivo_entrada \
               archivo_salida"<<endl;
        return 0;
    }
    cout<<"Iniciando lectura de archivo:"<<endl;

    Distribucion::EstablecerTamanoFrecuencias(24);
    Distribucion::EstablecerTamanoIntervalos(1);

    int stat= 0;
    string msg;
    unique_ptr<vector<Evento> > eventos;
    string archivo_entrada = argv[1];
    string archivo_salida = argv[2];

    ManejadorDeArchivosSecuencial manejador_de_archivos;
    stat = manejador_de_archivos.CargarDatos( archivo_entrada, eventos, msg);
    /*  ---
        Despu�s de realizar pruebas, se encontr� que el tiempo de ejecucion del
          cargue del archivo se ve afectado m�s por la velocidad de respuesta del
          disco que por el procesamiento de las l�neas y caracteres. Por lo tanto,
          se decidi� mantener este proceso secuencial
    */
    switch (stat)
    {
    case 0:
        cout<<"Lectura finalizada - total eventos: "<<eventos->size()<<endl;
        break;
    case -1:
        cout<<"El archivo "<<archivo_entrada<<" no se encontr�"<<endl;
        return 0;
        break;
    case -2:
        cout<<"Error al procesar el archivo:"<<endl<<msg<<endl;
        return 0;
        break;
    }
    AnalizadorDeDatosSecuencial analizador_de_datos;
    stat = analizador_de_datos.OrdenarEventos(eventos);
    /*  +++
        El ordenamiento se puede paralelizar con el algoritmo Merge Sort
    */

    switch (stat)
    {
    case 0:
        cout<<"Ordenamiento de eventos finalizado"<<endl;
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
        eventos, como las eventos de un grupo est�n cerca
        en el arreglo de eventos, los conflictos por varios hilos
        intentado actualizar el mismo grupo se ver�n minimizados.
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
        grupos, la �nica memoria que necesitan todos los hilos
        es la distribuci�n promedio, sin embargo esta s�lo es le�da,
        as� que no deber�a haber conflictos.
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
                                       ManejadorDeArchivosSecuencial::reemplazar);
    /*  ---
        No paralelizable - s�lo un hilo puede escribir a la vez
    */
    switch (stat)
    {
    case 0:
        cout<<"Generaci�n de archivo de salida finalizada"<<endl;
        break;
    case -1:
        cout<<"Error al generar archivo de salida:"<<endl<<msg<<endl;
        return 0;
        break;
    }

    return 0;
}
