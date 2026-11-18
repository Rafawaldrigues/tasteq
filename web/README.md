# tasteq web study

A public listening study built on the `tasteq` library. Each participant:

1. reads the consent page (anonymous code, 18+, can leave at any time) and picks
   a device and a study length;
2. sets a comfortable volume on a reference excerpt;
3. **part 1, screening:** answers A/B questions on synthetic sounds (no music),
   which teaches the model their general taste;
4. picks at least two music styles;
5. **part 2, music:** answers A/B questions on excerpts of songs from those
   styles, a few questions per song, at least three songs;
6. **part 3, blind test:** compares frozen EQs without being told which is
   which: learned vs flat, learned vs the EQ the model predicted for the same
   song right after part 1 (a snapshot stored with the participant, so this
   pair asks what training on the songs added) and learned vs
   a random EQ of the same strength. At least two songs in the blind test were
   never heard in part 2 (generalization). Blind answers never train the model;
7. fills a short questionnaire and sees their learned curves and blind score.

Every pair is loudness-matched (−23 LUFS, true peak ≤ −1 dBTP) by
`tasteq.dsp.render`, and the browser plays both versions in sync so switching
A/B compares the same moment.

## Files

| file | what |
|---|---|
| `study.yaml` | model kind, question counts per mode, blind-test pairs, audio format, devices, contact |
| `catalog.yaml` | the 28 FMA songs: title, artist, link, license, style, flags (explicit, AI-generated) |
| `prepare.py` | builds `template.sqlite`: synthetic sounds and all songs analyzed together |
| `study.py` | one participant's stage machine; the registry of participants |
| `app.py` | the HTTP API and the static front end (`static/`) |
| `export.py` | dumps every participant to CSV (and Parquet) for analysis |

Changing the model is one line in `study.yaml` (`model_kind: gp`), for any kind
registered in `tasteq.models.make_model`. The study only calls `Tuner.ask`,
`answer` and `best_gains`. Counts can be changed freely: each participant's plan
is stored when they start, so people already taking part are not affected.

## Data

Each participant has their own SQLite file in `$TASTEQ_DATA_DIR/participants/`,
copied from the template. Participants cannot interfere with each other, and
deleting someone's data (the "Leave, delete my answers" button, or a request
quoting their code) removes one file. The file holds the usual tasteq tables
(`sessions` with `kind` = `screening` / `music` / `blindtest`, `comparisons`,
`blind_trials`), a `web_answers` table (response time, plays of each version,
A/B switches, listening time) and the participant's state in `meta`. Partial
runs are kept: the export marks them `in_progress`, `abandoned` (inactive for
6 h) or `withdrawn`.

```bash
python -m web.export --out exports/             # participants, comparisons, blind_trials, answers
python -m web.export --out exports/ --parquet   # also Parquet (needs pyarrow)
```

Nothing identifies a participant: no name, e-mail or account, and the server
writes no access log, so IP addresses are not stored. The hosting platform's
own proxy logs are outside the app; check its retention settings.

## Run locally

```bash
pip install -e ".[ml,audio,web]"
export TASTEQ_MUSIC_DIR="/path/to/fma_lineup"   # the catalog's mp3 files
export TASTEQ_DATA_DIR="$PWD/web/data"          # gitignored
python -m web.prepare                           # once, ~1 min
uvicorn web.app:create_app --factory --port 8000
```

Then open <http://localhost:8000>. `python -m web.prepare --force` rebuilds the
template (after adding songs, for example); existing participants keep their
own copy.

## Deploy

The image is built from the repository root and expects a persistent volume at
`/data` with the songs in `/data/music`. On first start it builds the template
there, then serves on `$PORT`.

```bash
docker build -f web/Dockerfile -t tasteq-study .
docker run -p 8000:8000 -v "$PWD/study-data:/data" tasteq-study
```

Run **one** process (the Dockerfile does): participants are kept in memory
between requests. A small machine with 2 vCPU and 2 GB RAM handles a handful
of people at once; each GP question takes about a second of CPU.

- **Hugging Face Spaces** (Docker SDK): create a Space, enable persistent
  storage (mounted at `/data`), push this repository with the Dockerfile moved
  or copied to the root and `app_port: 8000` in the Space's README header.
  Upload the songs to `/data/music`.
- **Fly.io:** `fly launch --dockerfile web/Dockerfile --no-deploy`, then
  `fly volumes create study_data --size 3` and mount it at `/data` in
  `fly.toml`. Deploy, then copy the songs with `fly ssh sftp shell`
  (`put` into `/data/music`) and restart.
- **Render:** new Web Service from the repository, runtime Docker,
  Dockerfile path `web/Dockerfile`, and a disk mounted at `/data`. Upload the
  songs through the shell, then restart.

Before going public: set `contact.email` in `study.yaml`, check the music
licenses again (see the checklist in the project's lineup notes) and download
an export now and then as a backup.

## Credits

Songs from the [Free Music Archive](https://freemusicarchive.org/), each under
the license listed in `catalog.yaml` (CC BY 4.0 or CC0 1.0). The site shows the
credit, license link and a note that the excerpts are equalized (changed) next
to every song and on its credits page.
