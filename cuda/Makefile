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

projectName = clasificador_de_distribuciones_cuda
binOut = bin/$(uname_S)
binTest = bin/$(uname_S)_test
objOut = obj/$(uname_S)
libObjs = $(objOut)/util_cuda.o \
	$(objOut)/kernel.o \
	$(objOut)/analizador_de_datos_cuda.o \
	$(objOut)/manejador_de_archivos_cuda.o 
testObjs = $(objOut)/analizador_de_datos_cuda_test.o \
	$(objOut)/manejador_de_archivos_cuda_test.o \
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

$(libsComponentesCompartidos):componentescompartidoslib
	
componentescompartidoslib:
	$(MAKE) -C $(cmpCompartidos) lib

$(libsSecuencial): secuenciallib

secuenciallib:
	$(MAKE) -C $(secuencial) lib

# test objs
$(objOut)/%_test.o: test/%_test.cc
	g++ -Wall -O2 -I../ $(wGSLIncludes) $(wUTppIncludes) -c $< -o $@

$(objOut)/analizador_de_datos_cuda.o: implementacion/analizador_de_datos_cuda.h

$(objOut)/manejador_de_archivos_cuda.o: \
	implementacion/manejador_de_archivos_cuda.h

# lib objs
$(objOut)/%.o: implementacion/%.cc
	g++ -Wall -O2 -c -I../ $(wGSLIncludes) $< -o $@

$(objOut)/%.o: implementacion/%.cu
	nvcc --compiler-options -Wall -O2 -I../ $(wGSLIncludes) -dc $< -o $@

$(objOut)/analizador_de_datos_cuda.o: \
	implementacion/analizador_de_datos_cuda.h \
	implementacion/kernel.h \
	$(cmpCompartidos)/entidades.h \
	$(cmpCompartidos)/interfaces.h \
	$(impSecuencial)/analizador_de_datos_secuencial.h

$(objOut)/manejador_de_archivos_cuda.o: \
	implementacion/manejador_de_archivos_cuda.h \
	implementacion/kernel.h \
	$(cmpCompartidos)/entidades.h \
	$(cmpCompartidos)/interfaces.h \
	$(cmpCompartidos)/util_archivos.h \
	$(impSecuencial)/manejador_de_archivos_secuencial.h

$(objOut)/util_cuda.o: implementacion/util_cuda.h

$(objOut)/kernel.o: implementacion/kernel.h

# main
$(objOut)/%.o: %.cc
	g++ -Wall -O2 -I../ -c $< -o $@

# test main
$(testExe): $(testObjs) $(binLib)($(libObjs)) $(binLib)($(libsSecuencial)) \
		$(libsComponentesCompartidos)
	nvcc --compiler-options -Wall -O2 $(wUTppLibs) $(wGSLLibs) -o $@ $^ \
		-l:libUnitTest++.a -l:libgsl.a -l:libgslcblas.a

# lib
$(binLib)($(libObjs)): $(libObjs)

$(binLib)($(libsSecuencial)): $(libsSecuencial)

# exe
$(binExe): $(binLib)($(libObjs)) $(binLib)($(libsSecuencial)) \
		$(objOut)/main.o $(libsComponentesCompartidos)
	nvcc --compiler-options -Wall -O2 $(wGSLLibs) -o $@ $^ -l:libgsl.a \
		-l:libgslcblas.a

cleanall:
	$(RM) $(binLib) $(binExe) $(testExe) $(libObjs) $(testObjs) $(objOut)/main.o

cleantest: cleanall

cleanlib:
	$(RM) $(binLib) $(libObjs)

cleanexe: cleanlib
	$(RM) $(binExe) $(objOut)/main.o
