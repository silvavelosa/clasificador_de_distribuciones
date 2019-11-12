SHELL = /bin/sh

ifeq ($(OS),Windows_NT)
    uname_S := Windows
else
    uname_S := $(shell uname -s)
endif

ifeq ($(uname_S), Windows)
    wUTppIncludes = -IC:/Users/silva/Documents/programacion/unittest-cpp
    wUTppLibs = -LC:/Users/silva/Documents/programacion/unittest-cpp/Deliv/Release
    exeExt = .exe
endif

binOut = bin/$(uname_S)
binTest = bin/$(uname_S)_test
objOut = obj/$(uname_S)

$(info $(shell mkdir -p $(binOut) $(binTest) $(objOut)))

.PHONY: all test lib cleanall

all: $(binOut)/componentes_compartidos.a \
		$(binTest)/componentes_compartidos_test$(exeExt)

test: $(binTest)/componentes_compartidos_test$(exeExt)
	./$(binTest)/componentes_compartidos_test

lib: $(binOut)/componentes_compartidos.a

$(binTest)/componentes_compartidos_test$(exeExt): $(objOut)/entidades_test.o \
		test/main_test.cc
	g++ -Wall -O2 -I../ $(wUTppIncludes) -c test/main_test.cc \
		-o $(objOut)/main_test.o
	g++ $(wUTppLibs) -L$(binOut) \
		-o $(binTest)/componentes_compartidos_test$(exeExt) \
		$(objOut)/entidades_test.o $(objOut)/main_test.o \
		-l:libUnitTest++.a -l:componentes_compartidos.a

$(objOut)/entidades.o: entidades.cc entidades.h
	g++ -Wall -fopenmp -O2 -I../ -c entidades.cc \
		-o $(objOut)/entidades.o

$(binOut)/componentes_compartidos.a: $(objOut)/entidades.o
	ar -r -s $(binOut)/componentes_compartidos.a $(objOut)/entidades.o

$(objOut)/entidades_test.o: test/entidades_test.cc \
		$(binOut)/componentes_compartidos.a
	g++ -Wall -O2 -I../ $(wUTppIncludes) -c test/entidades_test.cc \
		-o obj/$(uname_S)/entidades_test.o

cleanall :
	rm $(binOut)/componentes_compartidos.a \
		$(objOut)/entidades.o $(objOut)/entidades_test.o \
		$(binTest)/componentes_compartidos_test$(exeExt)
