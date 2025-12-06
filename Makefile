CC=g++
CDEFINES=
SOURCES=src/main.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=shallenge_cl
LDFLAGS=-framework OpenCL
CFLAGS=-c -std=c++11 -Wall -O2

# OpenCL kernel source files (order matters - dependencies first)
CL_SOURCES=src/cl/sha256.cl src/cl/nonce.cl src/cl/shallenge.cl

all: src/kernel.h $(EXECUTABLE)

# Concatenate .cl files into single kernel source
src/kernel_combined.cl: $(CL_SOURCES)
	cat $^ > $@

# Generate header from combined kernel
src/kernel.h: src/kernel_combined.cl
	xxd -i $< > $@

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $(CDEFINES) $< -o $@

clean:
	rm -rf src/*.o src/kernel.h src/kernel_combined.cl $(EXECUTABLE)

.PHONY: all clean
