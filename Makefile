# Scorciatoie. Sul server: `make setup` una volta, poi `source ~/.local/mxagent/env.sh`.
PY ?= python3

.PHONY: setup test test-fast doctor warmup docker clean

setup:            ## toolchain completa nella home, senza sudo
	bash scripts/setup_server.sh

test:             ## test del framework e del golden model
	$(PY) -m pytest -q

test-fast:
	$(PY) -m pytest -q -k "not all_scale_pairs"

doctor:
	mxagent doctor

warmup:
	mxagent doctor --warmup --skip-llm

docker:
	docker build -f docker/Dockerfile -t mxagent .

clean:
	rm -rf chisel/build chisel/target chisel/project/target chisel/project/project .pytest_cache
