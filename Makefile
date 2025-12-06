CC=g++
CDEFINES=
SOURCES=src/main.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=shallenge_cl
LDFLAGS=-framework OpenCL
CFLAGS=-c -std=c++11 -Wall -O2

all: src/kernel.h $(EXECUTABLE)

src/kernel.h: src/kernel.cl
	xxd -i $< > $@

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $(CDEFINES) $< -o $@

clean:
	rm -rf src/*.o src/kernel.h $(EXECUTABLE)
