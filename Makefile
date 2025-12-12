CC=g++

# Configuration
DEFAULT_USERNAME ?= brandonros
# Tuned for ~100-500ms kernel dispatch (better interactivity + debugging)
GLOBAL_SIZE ?= 131072
LOCAL_SIZE ?= 64
HASHES_PER_THREAD ?= 16

CDEFINES=-DCL_TARGET_OPENCL_VERSION=300 \
         -DDEFAULT_USERNAME=\"$(DEFAULT_USERNAME)\" \
         -DGLOBAL_SIZE=$(GLOBAL_SIZE) \
         -DLOCAL_SIZE=$(LOCAL_SIZE) \
         -DHASHES_PER_THREAD=$(HASHES_PER_THREAD)

OUTDIR=output

# Platform-specific flags
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LDFLAGS=-framework OpenCL -pthread
else
    LDFLAGS=-lOpenCL -pthread
endif

CFLAGS=-c -std=c++17 -Wall -O2 -pthread

# Executables
SHALLENGE_EXE=$(OUTDIR)/shallenge_cl

all: $(OUTDIR) $(SHALLENGE_EXE)

$(OUTDIR):
	mkdir -p $(OUTDIR)

# Generate headers from kernel sources
$(OUTDIR)/kernel.h: src/shallenge.cl | $(OUTDIR)
	xxd -i $< > $@

# Shallenge (SHA-256 based)
$(SHALLENGE_EXE): $(OUTDIR)/main.o
	$(CC) $< $(LDFLAGS) -o $@

$(OUTDIR)/main.o: src/main.cpp $(OUTDIR)/kernel.h | $(OUTDIR)
	$(CC) $(CFLAGS) $(CDEFINES) -I$(OUTDIR) $< -o $@

clean:
	rm -rf $(OUTDIR)

.PHONY: all clean
