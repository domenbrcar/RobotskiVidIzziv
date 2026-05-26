# Robotski vid: analiza 9HPT videoposnetkov

Projekt obdela videoposnetke testa Nine Hole Peg Test. Program samodejno izbere ciljno mrežo, sledi roki, zazna zatiče in shrani pregledni GUI video, grafe ter trajektorije.

## Docker build

```bash
docker build -t rv-9hpt .
```

## Zagon na Windows

En video:

```powershell
docker run --rm -it --shm-size=16g `
  -v "${PWD}:/workspace" `
  -w /workspace `
  rv-9hpt python src/run.py `
  -video /workspace/data/patient_001/patient_001camP_0_20241121_10_21_17.mp4 `
  -output /workspace/output/final
```

En pacient:

```powershell
docker run --rm -it --shm-size=16g `
  -v "${PWD}:/workspace" `
  -w /workspace `
  rv-9hpt python src/run.py `
  -patient /workspace/data/patient_003 `
  -output /workspace/output/final
```

Vsi videi:

```powershell
docker run --rm -it --shm-size=16g `
  -v "${PWD}:/workspace" `
  -w /workspace `
  rv-9hpt python src/run.py `
  -folder /workspace/data `
  -output /workspace/output/final
```

## Zagon na Linux strežniku

```bash
docker run --rm -it \
  --shm-size=16g \
  -v "$(pwd):/workspace" \
  -w /workspace \
  rv-9hpt python src/run.py \
  -folder /workspace/data \
  -output /workspace/output/final
```

## Izhod

Za vsak video nastane samo:

```text
output/<run_name>/patient_XXX/<video_id>/
  gui_video/
    gui_<video_id>.mp4
  graphs/
    *.png
  trajectories/
    *.png
```
