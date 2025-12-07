CC=g++

# Configuration
DEFAULT_USERNAME ?= brandonros
GLOBAL_SIZE ?= 1048576
LOCAL_SIZE ?= 256
HASHES_PER_THREAD ?= 64

CDEFINES=-DCL_TARGET_OPENCL_VERSION=300 \
         -DDEFAULT_USERNAME=\"$(DEFAULT_USERNAME)\" \
         -DGLOBAL_SIZE=$(GLOBAL_SIZE) \
         -DLOCAL_SIZE=$(LOCAL_SIZE) \
         -DHASHES_PER_THREAD=$(HASHES_PER_THREAD)

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

CFLAGS=-c -std=c++17 -Wall -O2 -pthread

all: $(OUTDIR) $(OUTDIR)/kernel.h $(EXECUTABLE)

$(OUTDIR):
	mkdir -p $(OUTDIR)

# Generate header from kernel source
$(OUTDIR)/kernel.h: src/shallenge.cl | $(OUTDIR)
	xxd -i $< > $@

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

$(OUTDIR)/main.o: src/main.cpp $(OUTDIR)/kernel.h | $(OUTDIR)
	$(CC) $(CFLAGS) $(CDEFINES) -I$(OUTDIR) $< -o $@

clean:
	rm -rf $(OUTDIR)

.PHONY: all clean
