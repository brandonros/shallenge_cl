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

# OpenCL kernel source files (order matters - dependencies first)
CL_SOURCES=src/cl/sha256.cl src/cl/util.cl src/cl/nonce.cl src/cl/shallenge.cl

# Header dependencies
HEADERS=src/config.hpp \
        src/core/types.hpp \
        src/core/hash_utils.hpp \
        src/gpu/opencl_raii.hpp \
        src/gpu/device.hpp \
        src/gpu/context.hpp \
        src/mining/validator.hpp \
        src/mining/miner.hpp

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

$(OUTDIR)/main.o: src/main.cpp $(HEADERS) $(OUTDIR)/kernel.h | $(OUTDIR)
	$(CC) $(CFLAGS) $(CDEFINES) -Isrc -I$(OUTDIR) $< -o $@

clean:
	rm -rf $(OUTDIR)

# Test configuration
TEST_SOURCES=tests/test_main.cpp
TEST_EXECUTABLE=$(OUTDIR)/test_runner

test: $(TEST_EXECUTABLE)
	./$(TEST_EXECUTABLE)

$(TEST_EXECUTABLE): $(TEST_SOURCES) $(HEADERS) $(OUTDIR)/kernel.h | $(OUTDIR)
	$(CC) -std=c++17 -Wall -O2 $(CDEFINES) -Isrc -I$(OUTDIR) -Itests $(TEST_SOURCES) $(LDFLAGS) -o $@

.PHONY: all clean test
