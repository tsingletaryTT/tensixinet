# Common tasks. `make help` lists them.
.PHONY: help test mock demo clean

help:
	@echo "make test   run the test suite (no hardware needed)"
	@echo "make mock   render 20 s from 4 synthetic chips into audio/"
	@echo "make demo   record the split-pane demo with its soundtrack (needs tt-demo, tt-toplike, ffmpeg)"
	@echo "make clean  remove caches and generated audio and logs"

test:
	python3 -m pytest

mock:
	python3 -m tensixinet --mock-chips 4 --duration 20

demo:
	demo/build.sh

clean:
	rm -rf .pytest_cache audio logs
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
