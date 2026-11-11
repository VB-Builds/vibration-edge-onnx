# Deploying to ARM64 embedded Linux (unverified guide)

> **Status: unverified.** No ARM64 hardware was available; nothing here has been run
> on an ARM64 device. It is a sketch, so expect to adjust it. The container itself is
> only tested on x86-64 (Docker 29).

Many embedded Linux modules run applications as containers. This repo is already a
plain OCI image (`python:3.11-slim`, non-root, numpy + onnxruntime + paho-mqtt, all
with `linux/arm64` wheels), so the flow works on any ARM64 board or module that
runs a Docker-compatible container engine.

## 1. Build for arm64 on your PC

```bash
docker buildx create --use                      # once
docker buildx build --platform linux/arm64 \
  -t <registry>/<user>/vibration-edge-onnx:0.1 --push .
```

`--push` is needed because a multi-platform image cannot be loaded into the local
image store with the classic driver. Use `--platform linux/amd64,linux/arm64` for both.
Docs: <https://docs.docker.com/build/building/multi-platform/>

## 2. Get it onto the device

Either let the device pull from a registry, or save/load a tarball:

```bash
docker buildx build --platform linux/arm64 -o type=docker,dest=vib.tar \
  -t vibration-edge-onnx:0.1 .
scp vib.tar <user>@<device-ip>:   # then on the device: docker load -i vib.tar
```

## 3. Run it on the device

SSH in and use the container engine on the device, with the compose file from this
repo (point `image:` at your registry tag and drop the `build:` line):

```bash
docker compose up -d        # or: docker run -d --restart unless-stopped <image> --sim --loop --interval 1 --mqtt-host <broker>
```

For production, bake the image into the OS image or roll out updates as container
image updates from a registry, with a rollback path; I have not tried either.

## Things to check on real hardware

- Python wheels for your exact libc / arch (should be fine on Debian-based slim images).
- Replace the `--sim` source with a real accelerometer reader (SPI/I2C/ADC) feeding
  `edge.infer.run()` 3200-sample windows; the simulator is not a sensor driver.
- Per-window latency on the target SoC: run `python -m edge.infer --sim` and read the
  `latency` line on stderr.
- MQTT broker auth/TLS: the demo mosquitto config is anonymous and unencrypted.
