# atmos_quality

Unified render and measurement CLI for DAMF, E-AC-3 JOC and TrueHD.

## Install

```sh
python3 -m pip install -e .
cp .env.example .env   # edit paths
aq doctor
```

## Usage

```sh
aq encoder /path/to/master.mlp /path/to/candidate.ec3 --layouts 7.1.4,binaural
aq cluster /path/to/source.atmos /path/to/clustered.atmos --layouts 7.1.4
aq render /path/to/source.atmos --layout 7.1.4 --out /path/to/render.wav
```

Reports go to `$AQ_OUT/<run_id>/` (`report.json`, `report.md`, `metrics.csv`).
Cavern is built on demand. SPAUQ is the primary spatial metric; ViSQOL is optional. `aq doctor` shows both.

## SPAUQ

SPAUQ is GPL-3.0 and declares Python <3.11; it also works on 3.12 with:

```sh
python3 -m pip install fire
python3 -m pip install --ignore-requires-python 'git+https://github.com/kwatcharasupat/spauq.git@main'
```
