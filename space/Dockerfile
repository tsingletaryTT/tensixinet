# The tensixinet Hugging Face Space: synthetic chips, and the TT-Lang kernels run in the TT-Lang simulator. No Tenstorrent hardware.
#
# `tt-lang-sim` is the only provider of `ttnn` here (the simulator's own), which is what kernels/README.md asks for: the copy
# that comes with a full TT-Lang install can reach a real device. Torch is the CPU wheel, because a Space with no GPU has no use for CUDA.
# A Docker Space runs as user 1000 and listens on the port named by `app_port` in README.md.
FROM python:3.12-slim

# ffmpeg: Gradio's streaming audio needs it to cut each segment into the stream.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PORT=7860 PYTHONUNBUFFERED=1
WORKDIR /home/user/app

RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir tt-lang-sim==1.1.6 gradio==6.30.0 numpy soundfile

COPY --chown=user . .
RUN pip install --no-cache-dir --no-deps -e .

CMD ["python", "app/gradio_app.py"]
