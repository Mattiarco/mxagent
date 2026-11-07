#!/usr/bin/env bash
# Installa nella home, SENZA sudo, tutto ciò che serve a mxagent:
#
#   JDK 17 + sbt         tramite Coursier (binario `cs` del repo, o scaricato)
#   OSS CAD Suite        yosys, nextpnr-ecp5, verilator (build statiche ufficiali)
#   firtool 1.62.1       solo su aarch64: compilato dai sorgenti CIRCT (Chisel 6.6
#                        lo scarica già pronto solo per x86_64), vedi più sotto
#   venv Python          uv + Python fissato, mxagent + cocotb + matplotlib (.[sim,eval,dev])
#   cache sbt/Chisel     dipendenze Maven, plugin Chisel e firtool pre-scaricati,
#                        così le run successive funzionano anche offline
#
# Le versioni (sbt, firtool) sono lette da mxagent/versions.py: unica fonte.
#
# Architetture: Linux x86_64 e Linux aarch64 (es. DGX Spark). È idempotente:
# rilanciarlo salta ciò che è già presente. Uso:
#
#   bash scripts/setup_server.sh                     # installa in ~/.local/mxagent
#   bash scripts/setup_server.sh --prefix /scratch/me/mxagent
#   bash scripts/setup_server.sh --models qwen2.5-coder:14b,qwen2.5-coder:32b
#   bash scripts/setup_server.sh --add-to-bashrc     # aggiunge `source env.sh` a ~/.bashrc
#   bash scripts/setup_server.sh --skip-warmup       # niente elaborazione di prova
#   bash scripts/setup_server.sh --firtool-path DIR  # usa un firtool 1.62.1 già pronto
#   bash scripts/setup_server.sh --build-firtool     # compila firtool anche su x86_64
#   bash scripts/setup_server.sh --keep-build        # conserva la cartella di build
#
# Variabili: MXAGENT_BUILD_DIR (build di firtool, ~20 GB; default <prefix>/build),
# MXAGENT_BUILD_JOBS (job paralleli; default nproc), CC/CXX (compilatore).
#
# Alla fine:   source ~/.local/mxagent/env.sh   &&   mxagent doctor
set -euo pipefail

PREFIX="${HOME}/.local/mxagent"
MODELS=""
ADD_BASHRC=0
WARMUP=1
OSS_URL="${OSS_CAD_SUITE_URL:-}"   # forzabile, es. per una release precisa
JDK="${MXAGENT_JDK:-temurin:17}"
FIRTOOL_PATH_ARG=""
BUILD_FIRTOOL=0
KEEP_BUILD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --add-to-bashrc) ADD_BASHRC=1; shift ;;
    --skip-warmup) WARMUP=0; shift ;;
    --firtool-path) FIRTOOL_PATH_ARG="$2"; shift 2 ;;
    --build-firtool) BUILD_FIRTOOL=1; shift ;;
    --keep-build) KEEP_BUILD=1; shift ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) echo "opzione sconosciuta: $1" >&2; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# versioni bloccate: unica fonte mxagent/versions.py
pyver() { sed -n "s/^$1 *= *\"\([^\"]*\)\".*/\1/p" "${REPO}/mxagent/versions.py"; }
SBT_VERSION="$(pyver SBT_VERSION)"
FIRTOOL_VERSION="$(pyver FIRTOOL_VERSION)"
[[ -n "${SBT_VERSION}" && -n "${FIRTOOL_VERSION}" ]] || { echo "versioni non leggibili da mxagent/versions.py" >&2; exit 1; }
BIN="${PREFIX}/bin"
mkdir -p "${BIN}"
log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m  ✔\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m  !\033[0m %s\n' "$*"; }

# Architettura: decide i binari scaricati (Coursier, OSS CAD Suite, Ollama).
#   CS_ARCH    nome nell'asset di Coursier     cs-<CS_ARCH>-pc-linux.gz
#   OSS_ARCH   nome nell'asset di OSS CAD Suite oss-cad-suite-linux-<OSS_ARCH>-AAAAMMGG.tgz
#   OLLAMA_ARCH nome nell'archivio di Ollama   ollama-linux-<OLLAMA_ARCH>.tgz
#   ELF_MACHINE campo e_machine dell'header ELF (per riconoscere il `cs` del repo)
#   CMAKE_ARCH, NINJA_ASSET  binari di cmake/ninja, solo se si compila firtool
OS="$(uname -s)"
ARCH="$(uname -m)"
case "${OS}-${ARCH}" in
  Linux-x86_64|Linux-amd64)
    CS_ARCH="x86_64"; OSS_ARCH="x64";   OLLAMA_ARCH="amd64"; ELF_MACHINE="3e00"
    CMAKE_ARCH="x86_64";  NINJA_ASSET="ninja-linux.zip" ;;
  Linux-aarch64|Linux-arm64)
    CS_ARCH="aarch64"; OSS_ARCH="arm64"; OLLAMA_ARCH="arm64"; ELF_MACHINE="b700"
    CMAKE_ARCH="aarch64"; NINJA_ASSET="ninja-linux-aarch64.zip" ;;
  *)
    echo "piattaforma non supportata: ${OS}-${ARCH} (supportate: Linux x86_64, Linux aarch64)" >&2
    exit 1 ;;
esac
log "piattaforma ${OS}-${ARCH}"

# True se $1 è un eseguibile ELF a 64 bit per l'architettura di questa macchina.
# Legge l'header direttamente (`file` spesso non è installato sui server):
# byte 0-3 = \x7fELF, byte 4 = 02 (64 bit), byte 18-19 = e_machine little-endian.
elf_matches_host() {
  [[ -f "$1" ]] || return 1
  local magic cls mach
  magic="$(od -An -tx1 -N4 "$1" 2>/dev/null | tr -d ' \n')"
  cls="$(od -An -tx1 -j4 -N1 "$1" 2>/dev/null | tr -d ' \n')"
  mach="$(od -An -tx1 -j18 -N2 "$1" 2>/dev/null | tr -d ' \n')"
  [[ "${magic}" == "7f454c46" && "${cls}" == "02" && "${mach}" == "${ELF_MACHINE}" ]]
}

# ------------------------------------------------------------------ Coursier
CS="${BIN}/cs"
if [[ -x "${CS}" ]] && ! elf_matches_host "${CS}"; then
  warn "Coursier in ${CS} non è per ${ARCH}: lo riscarico"
  rm -f "${CS}"
fi
if [[ ! -x "${CS}" ]]; then
  log "Coursier: download cs-${CS_ARCH}-pc-linux.gz"
  curl -fL "https://github.com/coursier/coursier/releases/latest/download/cs-${CS_ARCH}-pc-linux.gz" \
    | gzip -d > "${CS}"
  chmod +x "${CS}"
fi
ok "coursier $("${CS}" version)"

# ------------------------------------------------------------- JDK 17 + sbt
log "JDK ${JDK} (Scala 2.13.14 / Chisel 6 sono testati su 17)"
JAVA_HOME_17="$("${CS}" java-home --jvm "${JDK}")"
ok "JAVA_HOME=${JAVA_HOME_17}"

# Launcher fissato alla stessa versione del progetto: con `cs install sbt` si
# otterrebbe l'ultimo (2.x), che funziona ma rende la toolchain non riproducibile.
if ! grep -aq "v${SBT_VERSION}" "${BIN}/sbt" 2>/dev/null; then
  log "sbt ${SBT_VERSION} (launcher + client sbtn)"
  "${CS}" install --install-dir "${BIN}" "sbt:${SBT_VERSION}" "sbtn:${SBT_VERSION}" >/dev/null
fi
ok "sbt ${SBT_VERSION} in ${BIN}"

# ------------------------------------------------------------ OSS CAD Suite
if [[ ! -x "${PREFIX}/oss-cad-suite/bin/yosys" ]]; then
  if [[ -z "${OSS_URL}" ]]; then
    OSS_URL="$(curl -fsSL https://api.github.com/repos/YosysHQ/oss-cad-suite-build/releases/latest \
      | grep -o "https://[^\"]*/oss-cad-suite-linux-${OSS_ARCH}-[0-9]\{8\}\.tgz" | head -1)"
  fi
  [[ -n "${OSS_URL}" ]] || {
    echo "URL di OSS CAD Suite per linux-${OSS_ARCH} non trovato nell'ultima release" \
         "(imposta OSS_CAD_SUITE_URL a mano)" >&2; exit 1; }
  log "OSS CAD Suite: ${OSS_URL}"
  tmp="$(mktemp -d)"
  curl -fL -o "${tmp}/oss.tgz" "${OSS_URL}"
  tar -xzf "${tmp}/oss.tgz" -C "${PREFIX}"
  rm -rf "${tmp}"
  echo "${OSS_URL}" > "${PREFIX}/oss-cad-suite/RELEASE_URL"
fi
ok "$("${PREFIX}/oss-cad-suite/bin/yosys" -V)"
ok "$("${PREFIX}/oss-cad-suite/bin/verilator" --version)"
ok "$("${PREFIX}/oss-cad-suite/bin/nextpnr-ecp5" --version 2>&1 | head -1)"

# ------------------------------------------------------------------ firtool
# Chisel 6.6 scarica firtool 1.62.1 da Maven (org.chipsalliance:llvm-firtool),
# che contiene binari solo per linux-x64, macos-x64 e windows-x64; nemmeno le
# release di CIRCT hanno binari linux-aarch64. Su aarch64 lo si compila qui dai
# sorgenti ufficiali della release (sha256 verificato), con gli stessi flag
# della build ufficiale (.github/workflows/unifiedBuildTestAndInstall.yml),
# e Chisel lo trova tramite CHISEL_FIRTOOL_PATH, che è la CARTELLA del binario.
# firtool-resolver NON controlla la versione del binario indicato: lo fa questo
# script (e `mxagent doctor`), perché un firtool diverso cambia il Verilog.
FIRTOOL_SHA256="827f071e898fa7e2f08bd91efae9e3bef9e8a9706a7c8ce235670d26a8578bf6"
CMAKE_VERSION="3.30.5"     # LLVM 18 richiede cmake >= 3.20
NINJA_VERSION="1.12.1"

# True se la cartella $1 contiene firtool ${FIRTOOL_VERSION}.
firtool_ok() {
  [[ -x "$1/firtool" ]] && "$1/firtool" --version 2>/dev/null | grep -qx "CIRCT firtool-${FIRTOOL_VERSION}"
}

# True se la versione $1 è >= $2.
version_ge() { [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" == "$2" ]]; }

# cmake >= 3.20 e ninja: quelli di sistema se adeguati, altrimenti binari
# ufficiali in <prefix>/tools (solo per la build, non finiscono in env.sh).
ensure_cmake_ninja() {
  local tools="${PREFIX}/tools" have
  have="$(cmake --version 2>/dev/null | sed -n '1s/[^0-9]*\([0-9.]*\).*/\1/p' || true)"
  if [[ -z "${have}" ]] || ! version_ge "${have}" 3.20; then
    if [[ ! -x "${tools}/cmake/bin/cmake" ]]; then
      log "cmake ${CMAKE_VERSION} (linux-${CMAKE_ARCH})"
      mkdir -p "${tools}/cmake"
      curl -fL "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-${CMAKE_ARCH}.tar.gz" \
        | tar -xz -C "${tools}/cmake" --strip-components=1
    fi
    export PATH="${tools}/cmake/bin:${PATH}"
  fi
  if ! command -v ninja >/dev/null 2>&1; then
    if [[ ! -x "${tools}/bin/ninja" ]]; then
      log "ninja ${NINJA_VERSION} (${NINJA_ASSET})"
      mkdir -p "${tools}/bin"
      curl -fL -o "${tools}/ninja.zip" \
        "https://github.com/ninja-build/ninja/releases/download/v${NINJA_VERSION}/${NINJA_ASSET}"
      # python3 invece di unzip, che sui server spesso manca
      python3 -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extract("ninja", sys.argv[2])' \
        "${tools}/ninja.zip" "${tools}/bin"
      chmod +x "${tools}/bin/ninja"
      rm -f "${tools}/ninja.zip"
    fi
    export PATH="${tools}/bin:${PATH}"
  fi
  ok "$(cmake --version | head -1), ninja $(ninja --version)"
}

# Compila firtool ${FIRTOOL_VERSION} e lo installa in $1 (binario in $1/bin).
# Riprendibile: rilanciando lo script, ninja riparte da dove si era fermato.
build_firtool() {
  local dest="$1"
  local work="${MXAGENT_BUILD_DIR:-${PREFIX}/build}/circt-${FIRTOOL_VERSION}"
  local jobs="${MXAGENT_BUILD_JOBS:-$(nproc)}"
  local cc cxx free_gb pid
  if command -v gcc >/dev/null 2>&1 && command -v g++ >/dev/null 2>&1; then cc=gcc; cxx=g++
  elif command -v clang >/dev/null 2>&1 && command -v clang++ >/dev/null 2>&1; then cc=clang; cxx=clang++
  else
    echo "per compilare firtool serve un compilatore C/C++ (gcc/g++ >= 7.4 o clang >= 5);" \
         "senza sudo: conda install -c conda-forge gxx, poi rilancia" >&2
    exit 1
  fi
  cc="${CC:-${cc}}"; cxx="${CXX:-${cxx}}"
  command -v git >/dev/null 2>&1 || { echo "per compilare firtool serve git (stringa di versione)" >&2; exit 1; }

  mkdir -p "${work}"
  free_gb=$(( $(df -Pk "${work}" | awk 'NR == 2 { print $4 }') / 1048576 ))
  if [[ ! -d "${work}/build" ]] && (( free_gb < 20 )); then
    echo "spazio insufficiente in ${work}: ${free_gb} GB liberi, ne servono ~20" \
         "(scegli un'altra cartella con MXAGENT_BUILD_DIR=...)" >&2
    exit 1
  fi
  ensure_cmake_ninja

  if [[ ! -f "${work}/src/CMakeLists.txt" ]]; then
    log "firtool ${FIRTOOL_VERSION}: sorgenti CIRCT (circt-full-sources.tar.gz, ~210 MB)"
    curl -fL -o "${work}/src.tgz" \
      "https://github.com/llvm/circt/releases/download/firtool-${FIRTOOL_VERSION}/circt-full-sources.tar.gz"
    echo "${FIRTOOL_SHA256}  ${work}/src.tgz" | sha256sum -c --quiet - \
      || { echo "sha256 dei sorgenti CIRCT non corrisponde: download corrotto?" >&2; exit 1; }
    rm -rf "${work}/src.part"
    mkdir -p "${work}/src.part"
    tar -xzf "${work}/src.tgz" -C "${work}/src.part"
    # firtool stampa "CIRCT <git describe --tags --match firtool*>", ma il tarball
    # non ha la storia git: senza questo stamperebbe "unknown git version".
    # Un repo con un solo commit vuoto e il tag della release basta; i sorgenti
    # restano non tracciati, quindi `--dirty` non scatta.
    (cd "${work}/src.part" \
      && git init -q \
      && git -c user.name=mxagent -c user.email=mxagent@localhost commit -q --allow-empty \
           -m "circt-full-sources firtool-${FIRTOOL_VERSION} (sha256 ${FIRTOOL_SHA256:0:16})" \
      && git tag "firtool-${FIRTOOL_VERSION}")
    mv "${work}/src.part" "${work}/src"
    rm -f "${work}/src.tgz"
  fi

  log "firtool ${FIRTOOL_VERSION}: configurazione (${cxx}, ${jobs} job)"
  # `-include cstdint`: LLVM 18 (base di firtool 1.62.1) usa uint64_t in
  # SmallVector.h & co. senza includere <cstdint>, che la libstdc++ di GCC >= 15
  # non porta più con sé ("'uint64_t' was not declared in this scope"). Forzare
  # l'header non tocca i sorgenti né il codice generato; con GCC più vecchi è innocuo.
  if ! cmake -G Ninja -S "${work}/src/llvm/llvm" -B "${work}/build" \
      -DCMAKE_CXX_FLAGS="${CXXFLAGS:-} -include cstdint" \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLVM_ENABLE_ASSERTIONS=OFF \
      -DLLVM_ENABLE_PROJECTS=mlir -DLLVM_EXTERNAL_PROJECTS=circt \
      -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR="${work}/src" -DLLVM_TARGETS_TO_BUILD=host \
      -DLLVM_BUILD_EXAMPLES=OFF -DLLVM_STATIC_LINK_CXX_STDLIB=ON -DLLVM_ENABLE_TERMINFO=OFF \
      -DLLVM_ENABLE_ZSTD=OFF -DLLVM_PARALLEL_LINK_JOBS=1 -DVERILATOR_DISABLE=ON \
      -DCIRCT_RELEASE_TAG_ENABLED=ON -DCIRCT_RELEASE_TAG=firtool \
      -DCMAKE_C_COMPILER="${cc}" -DCMAKE_CXX_COMPILER="${cxx}" \
      -DCMAKE_INSTALL_PREFIX="${dest}" > "${work}/configure.log" 2>&1; then
    tail -30 "${work}/configure.log" >&2
    echo "configurazione di firtool fallita (log completo: ${work}/configure.log)" >&2
    exit 1
  fi

  log "firtool ${FIRTOOL_VERSION}: compilazione di LLVM+MLIR+CIRCT, tipicamente 20-60 minuti"
  log "  log completo: ${work}/build.log"
  ninja -C "${work}/build" -j "${jobs}" install-firtool > "${work}/build.log" 2>&1 &
  pid=$!
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 60
    printf '  … %s\n' "$(grep -o '^\[[0-9]*/[0-9]*\]' "${work}/build.log" | tail -1)"
  done
  if ! wait "${pid}"; then
    tail -40 "${work}/build.log" >&2
    echo "compilazione di firtool fallita (log: ${work}/build.log). Rilanciando lo script" \
         "la build riprende da dove si è fermata." >&2
    exit 1
  fi
  if ! firtool_ok "${dest}/bin"; then
    echo "firtool compilato, ma non si dichiara ${FIRTOOL_VERSION}:" \
         "$("${dest}/bin/firtool" --version 2>&1 | grep CIRCT)" >&2
    exit 1
  fi
  if [[ ${KEEP_BUILD} -eq 0 ]]; then
    rm -rf "${work}"
  fi
}

FIRTOOL_BIN_DIR=""
if [[ -n "${FIRTOOL_PATH_ARG}" ]]; then
  if ! firtool_ok "${FIRTOOL_PATH_ARG}"; then
    echo "--firtool-path ${FIRTOOL_PATH_ARG}: serve la cartella che contiene firtool" \
         "${FIRTOOL_VERSION} (\`firtool --version\` deve stampare 'CIRCT firtool-${FIRTOOL_VERSION}')" >&2
    exit 1
  fi
  FIRTOOL_BIN_DIR="$(cd "${FIRTOOL_PATH_ARG}" && pwd)"
elif [[ "${CS_ARCH}" == "aarch64" || ${BUILD_FIRTOOL} -eq 1 ]]; then
  FIRTOOL_BIN_DIR="${PREFIX}/firtool-${FIRTOOL_VERSION}/bin"
  if ! firtool_ok "${FIRTOOL_BIN_DIR}"; then
    build_firtool "${PREFIX}/firtool-${FIRTOOL_VERSION}"
  fi
fi
if [[ -n "${FIRTOOL_BIN_DIR}" ]]; then
  ok "firtool ${FIRTOOL_VERSION} in ${FIRTOOL_BIN_DIR} (CHISEL_FIRTOOL_PATH)"
fi

# ------------------------------------------------------------------- env.sh
# NB: si espone solo oss-cad-suite/bin, NON il suo `environment` completo, che
# sostituirebbe anche python3 e altri tool di sistema.
cat > "${PREFIX}/env.sh" <<EOF
# generato da scripts/setup_server.sh — sorgente con: source ${PREFIX}/env.sh
export MXAGENT_PREFIX="${PREFIX}"
export JAVA_HOME="${JAVA_HOME_17}"
export PATH="${PREFIX}/venv/bin:${BIN}:\${JAVA_HOME}/bin:${PREFIX}/oss-cad-suite/bin:\${PATH}"
# Chisel/firtool e sbt scrivono qui le cache (riusate offline)
export COURSIER_CACHE="\${COURSIER_CACHE:-\${HOME}/.cache/coursier}"
EOF
if [[ -n "${FIRTOOL_BIN_DIR}" ]]; then
  cat >> "${PREFIX}/env.sh" <<EOF
# firtool ${FIRTOOL_VERSION} locale (Chisel 6.6 non ne distribuisce uno per ${ARCH})
export CHISEL_FIRTOOL_PATH="${FIRTOOL_BIN_DIR}"
EOF
fi
ok "ambiente in ${PREFIX}/env.sh"
if [[ ${ADD_BASHRC} -eq 1 ]] && ! grep -qF "${PREFIX}/env.sh" "${HOME}/.bashrc" 2>/dev/null; then
  echo "source ${PREFIX}/env.sh" >> "${HOME}/.bashrc"
  ok "aggiunto a ~/.bashrc"
fi
# shellcheck disable=SC1091
source "${PREFIX}/env.sh"

# ------------------------------------------------------------------- Python
# venv con uv (binario statico, niente sudo) e Python fissato in versions.py:
# non dipende dal python3 di sistema, che spesso non ha ensurepip/venv (pacchetto
# apt python3-venv) o è troppo nuovo per cocotb. Nel PATH esponiamo solo
# oss-cad-suite/bin, quindi il suo python non interferisce.
PYTHON_VERSION="$(pyver PYTHON_VERSION)"
UV="${BIN}/uv"
if [[ ! -x "${UV}" ]]; then
  log "uv (gestore di Python e venv)"
  curl -fL "https://github.com/astral-sh/uv/releases/latest/download/uv-${CS_ARCH}-unknown-linux-gnu.tar.gz" \
    | tar -xz -C "${BIN}" --strip-components=1
fi
export UV_PYTHON_INSTALL_DIR="${PREFIX}/python"
if [[ ! -x "${PREFIX}/venv/bin/python" ]]; then
  log "venv Python ${PYTHON_VERSION} in ${PREFIX}/venv"
  "${UV}" venv -q --python "${PYTHON_VERSION}" "${PREFIX}/venv"
fi
"${UV}" pip install -q --python "${PREFIX}/venv/bin/python" -e "${REPO}[sim,eval,dev]"
ok "$("${PREFIX}/venv/bin/mxagent" --version), cocotb $("${PREFIX}/venv/bin/python" -c 'import cocotb; print(cocotb.__version__)')"

# ------------------------------------------------------------------- Ollama
if command -v ollama >/dev/null 2>&1 || curl -fsS "${OLLAMA_HOST:-http://localhost:11434}/api/version" >/dev/null 2>&1; then
  ok "Ollama raggiungibile"
  if [[ -n "${MODELS}" ]]; then
    IFS=',' read -ra ms <<< "${MODELS}"
    for m in "${ms[@]}"; do
      log "ollama pull ${m}"
      ollama pull "${m}"
    done
  fi
else
  warn "Ollama non trovato. Senza sudo: scarica https://ollama.com/download/ollama-linux-${OLLAMA_ARCH}.tgz,"
  warn "estrailo in ${PREFIX}/ollama e avvia '${PREFIX}/ollama/bin/ollama serve' (es. in tmux)."
fi

# ------------------------------------------------- warm-up: cache sbt/Chisel
# Elabora il modulo Smoke: scarica Scala, Chisel, plugin e firtool nelle cache e
# verifica che il Verilog venga da firtool ${FIRTOOL_VERSION}. Poi funziona offline.
if [[ ${WARMUP} -eq 1 ]]; then
  log "warm-up: elaborazione Chisel di prova + controllo toolchain (qualche minuto la prima volta)"
  if (cd "${REPO}" && mxagent doctor --warmup --skip-llm); then
    ok "toolchain completa e cache popolate"
  else
    warn "doctor segnala problemi: vedi sopra"
    exit 1
  fi
fi

log "fatto. Ricorda:  source ${PREFIX}/env.sh   (poi: mxagent doctor)"
