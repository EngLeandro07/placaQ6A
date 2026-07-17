# =============================================================================
#  Makefile - qnn_infer (cliente nativo em C puro, API QNN direta)
#
#  Compilar NA PROPRIA PLACA (aarch64), com QAIRT_SDK_ROOT apontando pro SDK
#  la' instalado (ex.: export QAIRT_SDK_ROOT=$HOME/mctech/testePlaca/qairt/2.42.0.251225).
#  Nao precisa cross-compilar do container x86 - a placa tem gcc.
#
#  USO:
#     export QAIRT_SDK_ROOT=/caminho/pro/qairt/2.42.0.251225
#     make
#     ./qnn_infer modelo_int8.bin frame.raw
# =============================================================================

QAIRT_SDK_ROOT ?= $(HOME)/mctech/testePlaca/qairt/2.42.0.251225

# model.env e' a fonte unica de verdade (ver raiz do repo). O formato
# KEY=VALUE (sem espacos) e' valido tanto como Makefile quanto como shell
# env - por isso da' pra dar "include" direto aqui. Procura primeiro ao lado
# deste Makefile (onde deve ser copiado no deploy pra placa, ver README.md),
# senao na raiz do repo (build local, dentro do checkout completo).
-include ../model.env
-include model.env

# Fallback se nenhum model.env for encontrado (mesmo default do .c).
GRAPH_NAME ?= modelo_fp

CC      = gcc
CFLAGS  = -std=c11 -Wall -Wextra -O2 -I$(QAIRT_SDK_ROOT)/include/QNN \
          -DDEFAULT_GRAPH_NAME=\"$(GRAPH_NAME)\"
LDFLAGS = -ldl

qnn_infer: src/qnn_infer.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f qnn_infer

.PHONY: clean
