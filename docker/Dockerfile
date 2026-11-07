# Immagine riproducibile di mxagent (senza Vivado, che non è redistribuibile).
# Ollama gira FUORI dal container (sull'host con la GPU):
#
#   docker build -f docker/Dockerfile -t mxagent .
#   docker run --rm -it --add-host=host.docker.internal:host-gateway \
#       -v "$PWD/runs:/work/runs" mxagent doctor
#
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git gzip tar python3 python3-venv g++ make \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash mx
USER mx
WORKDIR /work
COPY --chown=mx:mx . /work

# Stesso script usato sui server: JDK 17, sbt, OSS CAD Suite, firtool (aarch64),
# venv Python, warm-up delle cache Chisel/firtool (poi funziona offline).
RUN bash scripts/setup_server.sh --prefix /home/mx/.local/mxagent

ENV OLLAMA_HOST=http://host.docker.internal:11434
ENTRYPOINT ["bash", "-c", "source /home/mx/.local/mxagent/env.sh && exec mxagent \"$@\"", "mxagent"]
CMD ["doctor"]
