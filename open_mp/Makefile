SHELL = /bin/sh
ARFLAGS = -r -s -v -U

ifeq ($(OS),Windows_NT)
    uname_S := Windows
else
    uname_S := $(shell uname -s)
endif

ifeq ($(uname_S), Windows)
	wUTppIncludes = -IC:/Users/silva/Documents/programacion/unittest-cpp
	wUTppLibs = \
		-LC:/Users/silva/Documents/programacion/unittest-cpp/Deliv/Release
	wGSLIncludes = -IC:/Users/silva/Documents/programacion/gsl-2.6
	wGSLLibs = -LC:/Users/silva/Documents/programacion/gsl-2.6/.libs \
		-LC:/Users/silva/Documents/programacion/gsl-2.6/cblas/.libs
    exeExt = .exe
endif

.SUFFIXES:
.SUFFIXES: .cc .o .a $(exeExt)

projectName = clasificador_de_distribuciones_open_mp
binOut = bin/$(uname_S)
binTest = bin/$(uname_S)_test
objOut = obj/$(uname_S)
libObjs = $(objOut)/analizador_de_datos_open_mp.o \
	$(objOut)/manejador_de_archivos_open_mp.o
testObjs = $(objOut)/analizador_de_datos_open_mp_test.o \
	$(objOut)/manejador_de_archivos_open_mp_test.o \
	$(objOut)/main_test.o
cmpCompartidos = ../componentes_compartidos
libsComponentesCompartidos = \
	$(cmpCompartidos)/$(binOut)/componentes_compartidos.a
secuencial = ../secuencial
impSecuencial = $(secuencial)/implementacion
libsSecuencial = \
	$(secuencial)/$(binOut)/clasificador_de_distribuciones_secuencial.a

testExe = $(binTest)/$(projectName)_test$(exeExt)
binLib = $(binOut)/$(projectName).a
binExe = $(binOut)/$(projectName)$(exeExt)

$(info $(shell mkdir -p $(binOut) $(binTest) $(objOut)))

.PHONY: all test lib exe cleanall cleantest cleanlib cleanexe secuenciallib \
		componentescompartidoslib

all: lib exe $(testExe)

test: $(testExe)
	./$(testExe)

lib: $(binLib)($(libObjs))

exe: $(binExe)

$(libsComponentesCompartidos): componentescompartidoslib

componentescompartidoslib:
	$(MAKE) -C $(cmpCompartidos) lib

$(libsSecuencial): secuenciallib

secuenciallib:
	$(MAKE) -C $(secuencial) lib

# test objs
$(objOut)/%_test.o: test/%_test.cc
	g++ -Wall -O2 -I../ $(wGSLIncludes) $(wUTppIncludes) -c $< -o $@

$(objOut)/analizador_de_datos_open_mp_test.o: \
		implementacion/analizador_de_datos_open_mp.h

$(objOut)/manejador_de_archivos_open_mp_test.o: \
		implementacion/manejador_de_archivos_open_mp.h

# lib objs
$(objOut)/%.o: implementacion/%.cc
	g++ -Wall -fopenmp -O2 -I../ $(wGSLIncludes) -c $< -o $@

$(objOut)/analizador_de_datos_open_mp.o: \
		implementacion/analizador_de_datos_open_mp.h \
		$(cmpCompartidos)/entidades.h \
		$(cmpCompartidos)/interfaces.h \
		$(impSecuencial)/analizador_de_datos_secuencial.h

$(objOut)/manejador_de_archivos_open_mp.o: \
		implementacion/manejador_de_archivos_open_mp.h \
		$(cmpCompartidos)/entidades.h \
		$(cmpCompartidos)/interfaces.h \
		$(cmpCompartidos)/util_archivos.h \
		$(impSecuencial)/manejador_de_archivos_secuencial.h

# main
$(objOut)/%.o: %.cc
	g++ -Wall -fopenmp -O2 -I../ -c $< -o $@

# test main
$(testExe): $(testObjs) $(binLib)($(libObjs)) $(binLib)($(libsSecuencial)) \
		$(libsComponentesCompartidos)
	g++ $(wUTppLibs) $(wGSLLibs) -fopenmp -o $@ $^ -l:libUnitTest++.a \
		-l:libgsl.a -l:libgslcblas.a

# lib
$(binLib)($(libObjs)): $(libObjs)

$(binLib)($(libsSecuencial)): $(libsSecuencial)

# exe
$(binExe): $(binLib)($(libObjs)) $(binLib)($(libsSecuencial)) \
		$(objOut)/main.o $(libsComponentesCompartidos)
	g++ $(wGSLLibs) -fopenmp -o $@ $^ -l:libgsl.a -l:libgslcblas.a

cleanall:
	$(RM) $(binLib) $(binExe) $(testExe) $(libObjs) $(testObjs) $(objOut)/main.o

cleantest: cleanall

cleanlib:
	$(RM) $(binLib) $(libObjs)

cleanexe: cleanlib
	$(RM) $(binExe) $(objOut)/main.o
