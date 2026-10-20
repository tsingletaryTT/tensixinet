# Common tasks. `make help` lists them.
.PHONY: help setup live test mock demo clean kernels-sim

help:
	@echo "make setup  build .venv (same as ./tensixinet.py --setup)"
	@echo "make live   play as it goes (same as ./tensixinet.py --live)"
	@echo "make test   run the test suite (no hardware needed)"
	@echo "make mock   render 20 s from 4 synthetic chips into audio/"
	@echo "make demo   record the split-pane demo with its soundtrack (needs tt-demo, tt-toplike, ffmpeg)"
	@echo "make kernels-sim  run every kernel in the TT-Lang simulator (set TT_LANG_SIM; no chip, no lease)"
	@echo "make clean  remove caches and generated audio and logs"

setup:
	./tensixinet.py --setup

live:
	./tensixinet.py --live

test:
	python3 -m pytest

mock:
	python3 -m tensixinet --mock-chips 4 --duration 20

demo:
	demo/build.sh

kernels-sim:
	@test -n "$(TT_LANG_SIM)" || (echo "set TT_LANG_SIM to a tt-lang-sim installed in an environment with no real ttnn" && exit 2)
	for k in kernels/tt*.py; do $(TT_LANG_SIM) $$k || exit 1; done

clean:
	rm -rf .pytest_cache audio logs
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
