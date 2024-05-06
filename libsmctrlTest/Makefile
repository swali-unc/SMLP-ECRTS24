# Makefile for compiling CUDA code with libsmctrl library

# Compiler
CC := gcc
NVCC := nvcc

# Flags
CFLAGS := -O2
LDFLAGS := -lsmctrl -L.

# Directories
SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin

# Files
SRC_FILES := $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRC_FILES))
EXEC := $(BIN_DIR)/a.out

# Targets
all: $(EXEC)

$(EXEC): $(OBJ_FILES) | $(BIN_DIR)
	$(NVCC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR) $(BIN_DIR):
	mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all clean
