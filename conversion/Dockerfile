# =============================================================================
#  Dockerfile - Ambiente de conversao de modelos para a Radxa Dragon Q6A
# =============================================================================
#
#  ESTRATEGIA: multi-stage build
#
#  Stage 1 (qairt_source): puxa a imagem ARM64 da Radxa APENAS para extrair
#  o SDK QAIRT 2.42. Nao roda nada nela - so copiamos o diretorio /root/qairt.
#
#  Stage 2 (final): imagem Ubuntu 24.04 x86_64 nativa. Recebe o SDK copiado
#  do stage 1. Como estamos em x86_64, os binarios de conversao do QAIRT
#  (qairt-converter, qairt-quantizer) e suas C extensions (.so) carregam
#  normalmente - sem QEMU, sem incompatibilidade de arquitetura.
#
#  POR QUE ISSO RESOLVE O PROBLEMA:
#  A imagem Radxa e' ARM64, mas o SDK QAIRT embutido nela tem as ferramentas
#  de conversao compiladas para x86_64 (pelo design do QAIRT: conversao roda
#  no PC host, inferencia roda na placa). Rodar o container ARM64 via QEMU
#  impede o Python ARM64 de carregar as C extensions x86_64. O multi-stage
#  contorna isso: usamos o x86_64 nativo como base e so importamos o SDK.
#
#  DOIS AMBIENTES PYTHON (mesmo racional do design original):
#  - /opt/venv-export  -> ultralytics + torch + onnx  (passos 01 e calib)
#  - Python do sistema -> QAIRT (SDK x86_64 nativo)   (passos 02, 03, 04)
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: fonte do QAIRT SDK.
# Puxamos a imagem ARM64 so para poder fazer COPY do /root/qairt.
# Nenhum RUN e' executado aqui - e' um stage de extracao puro.
# -----------------------------------------------------------------------------
FROM --platform=linux/arm64 radxazifeng278/qairt-npu-v68:v1.2 AS qairt_source

# -----------------------------------------------------------------------------
# Stage 2: imagem de trabalho x86_64.
# Ubuntu 22.04 (Jammy) porque as C extensions x86_64 do QAIRT foram compiladas
# contra libpython3.10 — e Ubuntu 22.04 traz Python 3.10 como padrao.
# -----------------------------------------------------------------------------
FROM --platform=linux/amd64 ubuntu:22.04

LABEL maintainer="eduardo"
LABEL description="Ambiente de conversao pt->onnx->dlc->quant->context-binary para Q6A (QCS6490, HTP v68)"

ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# 1. Copia o SDK QAIRT da imagem ARM64.
#    O /root/qairt contem binarios x86_64 para conversao E binarios ARM64
#    para inferencia na placa. Aqui usaremos apenas os x86_64.
# -----------------------------------------------------------------------------
COPY --from=qairt_source /root/qairt /root/qairt

# -----------------------------------------------------------------------------
# 2. Dependencias de sistema.
#    python3-numpy: necessario para o Python do sistema carregar os scripts
#    do QAIRT (qairt-converter etc. fazem 'import numpy' diretamente).
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-venv \
        python3-pip \
        libpython3.10 \
        libc++1 \
        libc++abi1 \
        libunwind8 \
        libatomic1 \
        libgl1 \
        libglib2.0-0 \
        git \
        wget \
    && rm -rf /var/lib/apt/lists/*

# Pacotes Python necessarios para os scripts do QAIRT (rodam no Python do sistema).
# Instalados via pip3 para garantir versoes compativeis com o SDK 2.42.
# onnx==1.12.0: o qairt-converter faz 'import onnx' para ler o .onnx, mas o
# SDK nao lista onnx como dependencia propria. Fixado em 1.12.0 porque versoes
# mais novas (1.13+) exigem protobuf>=3.20.2, incompativel com o
# protobuf==3.19.6 exigido pelo QAIRT.
RUN pip3 install --no-cache-dir \
    'numpy==1.26.4' \
    'pyyaml>=5.3' \
    'scipy>=1.10.1' \
    'six>=1.16.0' \
    'packaging>=24.0' \
    'protobuf==3.19.6' \
    'absl-py>=2.1.0' \
    'onnx==1.12.0'

# -----------------------------------------------------------------------------
# 3. Venv de EXPORT (ultralytics) - ISOLADO do QAIRT.
#    Para ALTERAR versoes: edite requirements-export.txt.
# -----------------------------------------------------------------------------
COPY requirements-export.txt /tmp/requirements-export.txt
RUN python3 -m venv /opt/venv-export \
    && /opt/venv-export/bin/pip install --upgrade pip \
    && /opt/venv-export/bin/pip install -r /tmp/requirements-export.txt

# -----------------------------------------------------------------------------
# 4. Workspace
# -----------------------------------------------------------------------------
WORKDIR /workspace

# Scripts e diretorios sao montados via -v no docker run (ver README).
CMD ["/bin/bash"]
