SHELL := /bin/bash
BIN := bin
SRC := face-detect.swift
BINARY := $(BIN)/face-detect
PREFIX := /opt/homebrew

.PHONY: all build test install uninstall models models-ir18 models-ir50 models-all models-status clean

all: build

build: $(BINARY)

$(BINARY): $(SRC)
	@mkdir -p $(BIN)
	swiftc -O -framework AVFoundation -framework CoreML -o $@ $<

test: build
	@./tests/run-tests.sh $(BINARY)

install: build
	install -d $(PREFIX)/bin
	install -m 755 $(BINARY) $(PREFIX)/bin/face-detect

uninstall:
	rm -f $(PREFIX)/bin/face-detect

models: models-ir18

models-ir18:
	@./scripts/install-models.sh ir18

models-ir50:
	@./scripts/install-models.sh ir50

models-all: models-ir18 models-ir50

models-status:
	@./scripts/install-models.sh status

clean:
	rm -rf $(BIN)
