# Robotski vid: analiza 9HPT videoposnetkov

Projekt obdela videoposnetke testa Nine Hole Peg Test (9HPT). Pipeline zaklene ciljno 3x3 mrežo, stabilizira izbrano pacientovo roko, izračuna kinematiko roke, palca in kazalca ter shrani slovenski GUI video in CSV izhode.

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

Predstavniški pregled po pacientih:

```bash
docker run --rm -it \
  --shm-size=16g \
  -v "$(pwd):/workspace" \
  -w /workspace \
  rv-9hpt python src/run.py \
  -folder /workspace/data \
  -output /workspace/output/representative \
  --one-per-patient
```

## Izhodne datoteke

Za vsak video nastane mapa:

```text
output/<run_name>/patient_XXX/<video_id>/
  gui_<video_id>.mp4
  gui_kinematics_<video_id>.csv
  gui_pin_states_<video_id>.csv
  gui_pin_events_<video_id>.csv
  gui_summary_<video_id>.csv
  gui_calibration_debug_<video_id>.json
```

Skupni pregled zagona je v:

```text
output/<run_name>/processing_report.csv
```

Število zatičev je trenutno število vizualno potrjenih zasedenih ROI v zaklenjeni ciljni mreži. Merilo gibanja je izračunano iz znanega razmika 32 mm med sosednjimi luknjami.
