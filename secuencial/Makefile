SHELL = /bin/sh
ARFLAGS = -r -s -v -U

ifeq ($(OS),Windows_NT)
    uname_S := Windows
else
    uname_S := $(shell uname -s)
endif

ifeq ($(uname_S), Windows)
	wUTppIncludes = -IC:/Users/silva/Documents/programacion/unittest-cpp
	wUTppLibs = -LC:/Users/silva/Documents/programacion/unittest-cpp/Deliv/Release
	wGSLIncludes = -IC:/Users/silva/Documents/programacion/gsl-2.6
	define wGSLLibs = -LC:/Users/silva/Documents/programacion/gsl-2.6/.libs \
		-LC:/Users/silva/Documents/programacion/gsl-2.6/cblas/.libs
	endef
    exeExt = .exe
endif

.SUFFIXES:
.SUFFIXES: .cc .o .a $(exeExt)

projectName = clasificador_de_distribuciones_secuencial
binOut = bin/$(uname_S)
binTest = bin/$(uname_S)_test
objOut = obj/$(uname_S)
libObjs = $(objOut)/analizador_de_datos_secuencial.o \
	$(objOut)/manejador_de_archivos_secuencial.o
testObjs = $(objOut)/analizador_de_datos_secuencial_test.o \
	$(objOut)/manejador_de_archivos_secuencial_test.o \
	$(objOut)/main_test.o
cmpCompartidos = ../componentes_compartidos
libsComponentesCompartidos = $(cmpCompartidos)/$(binOut)/componentes_compartidos.a

testExe = $(binTest)/$(projectName)_test$(exeExt)
binLib = $(binOut)/$(projectName).a
binExe = $(binOut)/$(projectName)$(exeExt)

$(info $(shell mkdir -p $(binOut) $(binTest) $(objOut)))

.PHONY: all test lib exe cleanall cleantest cleanlib cleanexe

all: lib exe $(testExe)

test: $(testExe)
	./$(testExe)

lib: $(binLib)($(libObjs))

exe: $(binExe)

$(libsComponentesCompartidos):
	$(MAKE) -C $(cmpCompartidos)

# test objs
$(objOut)/%_test.o: test/%_test.cc
	g++ -Wall -O2 -I../ $(wUTppIncludes) -c $< -o $@

# lib objs 
$(objOut)/%.o: implementacion/%.cc
	g++ -Wall -O2 -I../ -c $< -o $@

# main
$(objOut)/%.o: %.cc
	g++ -Wall -O2 -I../ -c $< -o $@

# test main
$(testExe): $(testObjs) $(binLib)($(libObjs)) $(libsComponentesCompartidos)
	g++ $(wUTppLibs) $(wGSLLibs) -o $@ $^ -l:libUnitTest++.a \
		-l:libgsl.a -l:libgslcblas.a

# lib
$(binLib)($(libObjs)): $(libObjs)

# exe
$(binExe): $(binLib)($(libObjs)) $(objOut)/main.o $(libsComponentesCompartidos)
	g++ $(wGSLLibs) -o $@ $^ -l:libgsl.a -l:libgslcblas.a

cleanall:
	$(RM) $(binLib) $(binExe) $(testExe) $(libObjs) $(testObjs) 

cleantest: cleanall

cleanlib:
	$(RM) $(binLib) $(libObjs)

cleanexe: cleanlib 
	$(RM) $(binExe) $(objOut)/main.o
