# Orin Docker Refactor 2026-06

## Goal

Reduce rebuild time for the Jetson Orin Docker workflow when ROS Noetic and flash-attn are both enabled.

The original path built everything inside a single image:

- base system and Python runtime
- PyTorch / torchvision / Triton
- flash-attn 2.5.5 SM87
- ROS Noetic from source
- FluxVLA Python package

That made retries expensive. A transient network failure during ROS build could force a full flash-attn rebuild on the next attempt.

## Refactor Plan

The refactor follows three steps.

### Step 1. Flash-attn wheel cache

Add a standalone wheel build target in the unified build script:

- `docker/build_docker.sh wheel`

This script builds a reusable `flash-attn 2.5.5` SM87 wheel under:

```text
/mnt/nvme/fluxvla-wheels
```

The FA image build now prefers installing from a staged wheel in `docker/.wheelhouse/` and only falls back to source compilation when no wheel is available.

### Step 2. Split image targets by responsibility

Use one multi-stage Dockerfile and manage the staged targets through the unified build script:

- `docker/Dockerfile.orin --target base`
- `docker/Dockerfile.orin --target fa`
- `docker/Dockerfile.orin --target ros`
- `docker/Dockerfile.orin --target ros-fa`
- `docker/build_docker.sh base`
- `docker/build_docker.sh fa`
- `docker/build_docker.sh ros`
- `docker/build_docker.sh ros-fa`

Produced image layout:

```text
fluxvla:orin-base
fluxvla:orin-fa
fluxvla:orin-ros
fluxvla:orin-ros-fa
```

This isolates the expensive parts:

- changing FA no longer forces a ROS rebuild
- changing ROS no longer forces an FA rebuild
- retries can resume from a narrower layer boundary

### Step 3. Reuse ROS as a separate image layer

The `ros-fa` Dockerfile target does not rebuild ROS. It copies `/opt/ros/noetic` from the `ros` stage:

```text
COPY --from=ros_layer /opt/ros/noetic /opt/ros/noetic
```

That turns the combined ROS+FA image into a cheap composition step once both inputs already exist.

## New Build Flow

### One-time or rare rebuilds

```bash
cd FluxVLA
docker/build_docker.sh
```

### Typical rebuild scenarios

Only flash-attn changed:

```bash
docker/build_docker.sh wheel
docker/build_docker.sh fa
docker/build_docker.sh ros-fa
```

Only ROS changed:

```bash
docker/build_docker.sh ros
docker/build_docker.sh ros-fa
```

Only Python/FluxVLA base dependencies changed:

```bash
docker/build_docker.sh all
```

## Mirror Support

The unified build entrypoint keeps the existing optional mirror support:

```bash
FLUXVLA_USE_CN_MIRRORS=1 docker/build_docker.sh all
FLUXVLA_USE_CN_MIRRORS=1 docker/build_docker.sh ros
```

Custom mirrors still work through:

- `FLUXVLA_UBUNTU_PORTS_MIRROR`
- `FLUXVLA_PIP_INDEX_URL`
- `FLUXVLA_PIP_TRUSTED_HOST`

## Compatibility Notes

- `docker/run_docker.sh` defaults to the recommended `fluxvla:orin-ros-fa` image.
- The refactor does not change runtime semantics by itself; it only changes how images are built and reused.

## Expected Benefits

### Before

- `flash-attn + ROS` were coupled in one long build
- a ROS network failure could waste a previous FA compile
- a FA rebuild could waste a previously successful ROS build

### After

- flash-attn can be reused as a wheel
- ROS can be reused as a dedicated image
- the final `orin-ros-fa` image becomes mostly a composition step
- retries become much cheaper and more targeted

## Current Result

The repository now contains:

- reusable FA wheel build path
- one multi-stage `docker/Dockerfile.orin` with base / FA / ROS / ROS+FA targets
- unified build entrypoint `docker/build_docker.sh`
- process documentation in this file

The next practical step is to validate the unified build entrypoint end-to-end on the Orin host.