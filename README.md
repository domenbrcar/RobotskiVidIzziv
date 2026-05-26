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

Na Linux strežniku so podatki na disku:

```text
/media/FastDataMama/data_rv_26
```

Zato jih v Docker priklopimo kot `/data`.

En video:

```bash
docker run --rm -it \
  --shm-size=16g \
  -v "$(pwd):/workspace" \
  -v /media/FastDataMama/data_rv_26:/data:ro \
  -w /workspace \
  rv-9hpt python src/run.py \
  -video /data/patient_001/patient_001camP_0_20241121_10_21_17.mp4 \
  -output /workspace/output/final
```

En pacient:

```bash
docker run --rm -it \
  --shm-size=16g \
  -v "$(pwd):/workspace" \
  -v /media/FastDataMama/data_rv_26:/data:ro \
  -w /workspace \
  rv-9hpt python src/run.py \
  -patient /data/patient_003 \
  -output /workspace/output/final
```

Vsi videi:

```bash
docker run --rm -it \
  --shm-size=16g \
  -v "$(pwd):/workspace" \
  -v /media/FastDataMama/data_rv_26:/data:ro \
  -w /workspace \
  rv-9hpt python src/run.py \
  -folder /data \
  -output /workspace/output/final
```