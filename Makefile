CC=g++
CDEFINES=-DCL_TARGET_OPENCL_VERSION=300
SOURCES=src/main.cpp
OUTDIR=output
OBJECTS=$(OUTDIR)/main.o
EXECUTABLE=$(OUTDIR)/shallenge_cl

# Platform-specific flags
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LDFLAGS=-framework OpenCL -pthread
else
    LDFLAGS=-lOpenCL -pthread
endif

CFLAGS=-c -std=c++11 -Wall -O2 -pthread

# OpenCL kernel source files (order matters - dependencies first)
CL_SOURCES=src/cl/sha256.cl src/cl/util.cl src/cl/nonce.cl src/cl/shallenge.cl

all: $(OUTDIR) $(OUTDIR)/kernel.h $(EXECUTABLE)

$(OUTDIR):
	mkdir -p $(OUTDIR)

# Concatenate .cl files into single kernel source
$(OUTDIR)/kernel_combined.cl: $(CL_SOURCES) | $(OUTDIR)
	cat $^ > $@

# Generate header from combined kernel
$(OUTDIR)/kernel.h: $(OUTDIR)/kernel_combined.cl
	xxd -i $< > $@

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

$(OUTDIR)/main.o: src/main.cpp $(OUTDIR)/kernel.h | $(OUTDIR)
	$(CC) $(CFLAGS) $(CDEFINES) -I$(OUTDIR) $< -o $@

clean:
	rm -rf $(OUTDIR)

.PHONY: all clean
