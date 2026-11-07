"""Versioni bloccate della toolchain: unica fonte, letta da doctor, setup e Dockerfile.

Chisel 6.6.0 è pubblicato con firtool 1.62.1 (firtool-resolver lo scarica da
Maven per linux-x64, macos-x64 e windows-x64; su aarch64 va compilato, vedi
scripts/setup_server.sh). Un firtool diverso cambia il Verilog emesso.
"""
PYTHON_VERSION = "3.12"  # venv creato da setup_server.sh con uv
SCALA_VERSION = "2.13.14"
SBT_VERSION = "1.10.1"
CHISEL_VERSION = "6.6.0"
FIRTOOL_VERSION = "1.62.1"
JDK_RECOMMENDED = 17
JDK_RANGE = (11, 21)

#: Verilator minimo richiesto da ciascuna serie di cocotb (versione cocotb ->
#: Verilator minimo). Da riconfermare sul server in M2 con il test di fumo cocotb.
COCOTB_VERILATOR_MIN = {
    1: (5, 6),    # cocotb 1.9.x
    2: (5, 36),   # cocotb 2.0.x
}
VERILATOR_MIN = (5, 6)
