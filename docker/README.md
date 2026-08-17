# Dockerfiles of verl

We provide pre-built Docker images for quick setup. And from this version, we utilize a new image release hierarchy for productivity and stability.

Start from v0.6.0, we use vllm and sglang release image as our base image.

Start from v0.7.0, since vllm/vllm-openai:v0.12.0 is a minimal image without some essential libraries, we use nvidia/cuda:12.9.1-devel-ubuntu22.04 as our base image for vllm.

## Base Image

- vLLM: https://hub.docker.com/r/nvidia/cuda
- SGLang: https://hub.docker.com/r/lmsysorg/sglang

## Application Image

Upon base image, the following packages are added:
- flash_attn
- Megatron-LM
- Apex
- TransformerEngine
- DeepEP

Latest docker file:
- [Dockerfile.stable.vllm](https://github.com/verl-project/verl/blob/main/docker/Dockerfile.stable.vllm)
- [Dockerfile.stable.sglang](https://github.com/verl-project/verl/blob/main/docker/Dockerfile.stable.sglang)

All pre-built images are available in dockerhub: https://hub.docker.com/r/verlai/verl. For example, `verlai/verl:sgl0512.latest`, `verlai/verl:vllm024.latest`.

You can find the latest images used for development and ci in our github workflows:
- [.github/workflows/vllm.yml](https://github.com/verl-project/verl/blob/main/.github/workflows/vllm.yml)
- [.github/workflows/sgl.yml](https://github.com/verl-project/verl/blob/main/.github/workflows/sgl.yml)


## Building Locally

To build an image from source:

```sh
docker build -f docker/Dockerfile.stable.vllm -t verl:vllm-local .
```

For users in China who need an apt mirror to speed up package downloads, pass `APT_MIRROR`:

```sh
docker build -f docker/Dockerfile.stable.vllm \
    --build-arg APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn \
    -t verl:vllm-local .
```

### GB200 / aarch64

Pre-built images for GB200 (aarch64) are not yet published. Users should build locally on an aarch64 machine. Pre-built images will be added once available. This applies to the uv image below too — it supports aarch64 from the same Dockerfile and the same `uv.lock`.

```sh
docker build -f docker/Dockerfile.stable.vllm -t verl:vllm-arm64 .
docker build -f docker/Dockerfile.uv.cu130 -t verl:uv-cu130-arm64 .   # uv image
```

Cross-building from an x86_64 machine is possible but slow — `docker buildx build --platform linux/arm64` (after `docker run --privileged --rm tonistiigi/binfmt --install arm64`) runs the apt install, the GDRCopy build and the `prefetch` stage's `megatron-core` / `mbridge` source builds under QEMU. If your arm64 machine is remote, attach it as a *native* buildx node instead and drive the build from x86_64 with no emulation:

```sh
docker buildx create --name verl-arm --driver docker-container \
    --platform linux/arm64 ssh://user@arm-host --use
docker buildx build --platform linux/arm64 -f docker/Dockerfile.uv.cu130 \
    -t verl:uv-cu130-arm64 --load .
```

### One tag for both arches

Both arches can and should share a single tag, published as a multi-arch
manifest list — `docker pull verlai/verl:uv.cu130` then resolves to whichever
arch the puller is on, and `FROM verlai/verl:uv.cu130` in a downstream image
resolves per build platform. This is how `nvidia/cuda:13.0.2-devel-ubuntu24.04`
itself works, and what lets one Dockerfile and one job spec cover both.

If a single builder can reach both platforms (e.g. a native x86_64 node plus a
native arm64 node in the same builder), one command does it:

```sh
docker buildx build --platform linux/amd64,linux/arm64 \
    -f docker/Dockerfile.uv.cu130 -t verlai/verl:uv.cu130 --push .
```

More often the two builds happen on different machines at different times. Then
push arch-suffixed tags and join them afterwards — the join is metadata only, so
it is instant and re-runnable:

```sh
# on each host, natively:
docker buildx build -f docker/Dockerfile.uv.cu130 -t verlai/verl:uv.cu130-amd64 --push .
docker buildx build -f docker/Dockerfile.uv.cu130 -t verlai/verl:uv.cu130-arm64 --push .

# then, from anywhere:
docker buildx imagetools create -t verlai/verl:uv.cu130 \
    verlai/verl:uv.cu130-amd64 verlai/verl:uv.cu130-arm64
docker buildx imagetools inspect verlai/verl:uv.cu130   # verify both platforms
```

Three things to know:

- A multi-platform build **must** `--push`; `--load` cannot put an index into
  the local docker image store (load one arch at a time instead).
- Add `--provenance=false` if your registry or runtime trips over the extra
  `unknown/unknown` attestation entries buildx puts in the index by default.
- Pin by **tag**, not digest, to keep the multi-arch behaviour — a digest names
  one specific manifest, so `FROM ...@sha256:...` pins you to a single arch.

## uv image (`Dockerfile.uv.cu130`)

`docker/Dockerfile.uv.cu130` builds one image around verl's universal
`uv.lock` (GPU: CUDA 13.0 / torch 2.11 — vllm, sglang, fsdp, megatron — plus
the GPU-free `cpu` slice), on x86_64 and aarch64 alike. The build **bakes the
full uv package cache for every backend** into the image (the `prefetch`
stage); it does *not* bake a fixed `.venv`. Build with BuildKit:

```sh
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.uv.cu130 -t verl:uv-cu130 .
```

You pick the backend combination at **run** time (not build time) by syncing it
yourself (it must be conflict-free; see `[tool.uv].conflicts`). The container
starts in a shell — `manage_envs.py sync` `uv sync`s the requested extras into
`/workspace/verl/.venv` from the baked cache (fast / offline), and that venv is
already on `PATH`:

```sh
docker run --rm -it --gpus all verl:uv-cu130 bash
# then, inside the container:
python3 manage_envs.py sync sglang megatron -- --frozen
python3 -m verl.trainer.main_ppo ...
```

Optional named stages: `--target=prefetch` builds just the baked cache (every
backend, no source), and `--target=lock` regenerates `uv.lock`. A companion
`docker/Dockerfile.uv.cu129` builds the cu12.9 / torch-2.9.1 backends (veomni,
nemoautomodel) the same way; trtllm stays deferred. For the full story — the
manual sync flow, the baked-cache mechanics, and re-locking — see the
**"Install from the uv images"** section in
[`docs/start/install.rst`](../docs/start/install.rst).

## Installation from Docker

After pulling the desired Docker image and installing desired inference and training frameworks, you can run it with the following steps:

1. Launch the desired Docker image and attach into it:

```sh
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" --cap-add=SYS_ADMIN -v .:/workspace/verl --name verl <image:tag> sleep infinity
docker start verl
docker exec -it verl bash
```

2. If you use the images provided, you only need to install verl itself without dependencies:

```sh
# install the nightly version (recommended)
git clone https://github.com/verl-project/verl && cd verl
pip3 install --no-deps -e .
```

[Optional] If you hope to switch between different frameworks, you can install verl with the following command:

```sh
# install the nightly version (recommended)
git clone https://github.com/verl-project/verl && cd verl
pip3 install -e .[vllm]
pip3 install -e .[sglang]
```

## Release History

- 2026/07/21: update vllm stable image to vllm==0.24.0 (torch==2.11.0, CUDA 13.0.2, Ubuntu 24.04); update sglang stable image to sglang==0.5.12
- 2026/03/10: update vllm stable image to vllm==0.17.0; update sglang stable image to sglang==0.5.9
- 2026/01/17: update vllm stable image to torch==2.9.1, cudnn==9.16, deepep==1.2.1
- 2025/12/23: update vllm stable image to vllm==0.12.0; update sglang stable image to sglang==0.5.6
- 2025/11/18: update vllm stable image to vllm==0.11.1; update sglang stable image to sglang==0.5.5

