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
    exeExt = .exe
endif

.SUFFIXES:
.SUFFIXES: .cc .o .a $(exeExt)

binOut = bin/$(uname_S)
binTest = bin/$(uname_S)_test
objOut = obj/$(uname_S)
libObjs = $(objOut)/entidades.o

$(info $(shell mkdir -p $(binOut) $(binTest) $(objOut)))

.PHONY: all test lib cleanall cleantest cleanlib

all: lib $(binTest)/componentes_compartidos_test$(exeExt)

test: $(binTest)/componentes_compartidos_test$(exeExt)
	./$(binTest)/componentes_compartidos_test

lib: $(binOut)/componentes_compartidos.a($(libObjs))

$(objOut)/%_test.o: test/%_test.cc
	g++ -Wall -O2 -I../ $(wUTppIncludes) -c $< -o $@

$(objOut)/%.o: %.cc
	g++ -Wall -fopenmp -O2 -I../ -c $< -o $@

$(binTest)/componentes_compartidos_test$(exeExt): $(objOut)/entidades_test.o \
		$(objOut)/main_test.o $(binOut)/componentes_compartidos.a($(libObjs))
	g++ $(wUTppLibs) -o $@ $^ -l:libUnitTest++.a

$(binOut)/componentes_compartidos.a($(objOut)/entidades.o): $(objOut)/entidades.o

cleanall:
	$(RM) $(binOut)/componentes_compartidos.a \
		$(objOut)/entidades.o $(objOut)/entidades_test.o \
		$(binTest)/componentes_compartidos_test$(exeExt)

cleantest: cleanall

cleanlib:
	$(RM) $(binOut)/componentes_compartidos.a $(objOut)/entidades.o

