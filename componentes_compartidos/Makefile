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

projectName = componentes_compartidos
binOut = bin/$(uname_S)
binTest = bin/$(uname_S)_test
objOut = obj/$(uname_S)
libObjs = $(objOut)/entidades.o $(objOut)/util_archivos.o
testObjs = $(objOut)/entidades_test.o $(objOut)/main_test.o

testExe = $(binTest)/$(projectName)_test$(exeExt)
binLib = $(binOut)/$(projectName).a

$(info $(shell mkdir -p $(binOut) $(binTest) $(objOut)))

.PHONY: all test lib cleanall cleantest cleanlib

all: lib $(testExe)

test: $(testExe)
	./$(testExe)

lib: $(binLib)($(libObjs))

$(objOut)/%_test.o: test/%_test.cc
	g++ -Wall -O2 -I../ $(wUTppIncludes) -c $< -o $@

$(objOut)/%.o: %.cc
	g++ -Wall -fopenmp -O2 -I../ -c $< -o $@

$(objOut)/entidades.o: entidades.h

$(testExe): $(testObjs) $(binLib)($(libObjs))
	g++ $(wUTppLibs) -o $@ $^ -l:libUnitTest++.a

$(binLib)($(libObjs)): $(libObjs)

cleanall:
	$(RM) $(binLib) $(libObjs) $(testExe) $(testObjs)

cleantest: cleanall

cleanlib:
	$(RM) $(binLib) $(libObjs)

