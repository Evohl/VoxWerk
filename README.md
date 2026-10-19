# VoxWerk

VoxWerk is a local-first speech studio for browsing Piper voices, downloading only the models you choose, and generating speech on your own computer. The application package contains no voice models, recordings, training data, or machine-specific settings.

## Features

- Browse and filter the Piper voice catalog by language, quality, and name.
- Download individual models on demand with catalog-provided size and checksum verification.
- Enter text, adjust speaking rate, preview speech, and save WAV recordings.
- Select browser audio input and output devices, test the microphone level, and play an output test tone.
- Keep models and recordings in the user's data directory; keep device choices in browser-local storage.
- Run the web interface on loopback by default.

## Requirements

- Python 3.10 or newer.
- An internet connection to load the Piper catalog and download selected voice models.
- A browser with JavaScript enabled. Microphone and output-device selection support varies by browser; microphone labels may require permission.

## Install

From a source checkout:

```sh
python -m pip install .
```

Or install directly from GitHub:

```sh
python -m pip install "git+https://github.com/Evohl/VoxWerk.git"
```

Piper is installed as a runtime dependency. The first installation does not download any voice models.

## Run

```sh
voxwerk
```

Open <http://127.0.0.1:8765>. The server listens on loopback only by default. To use another local port:

```sh
voxwerk --port 8766
```

The server also accepts `--host`, but binding to a non-loopback address exposes the interface to other machines. Do this only on a trusted network and with appropriate network controls.

Open **Stimmen**, filter the catalog, and download a model. Then return to **Studio**, choose the installed model, enter text, and preview or save the generated WAV. Audio devices can be selected and tested in **Einstellungen**. VoxWerk requests microphone access only after an explicit microphone-test or device-discovery action.

## Data and privacy

- Voice models and WAV recordings are stored under `${XDG_DATA_HOME:-~/.local/share}/voxwerk`.
- The Piper catalog cache is stored under `${XDG_CACHE_HOME:-~/.cache}/voxwerk`.
- Audio device selections are stored in the current browser's local storage.
- Text submitted for speech synthesis is sent to the local VoxWerk server and is not uploaded by VoxWerk to a speech service.
- Catalog browsing and model downloads contact Hugging Face at `rhasspy/piper-voices`.
- The catalog and selected model files are checked against catalog metadata before downloaded files are activated.

Deleting the VoxWerk data directory removes downloaded voices and saved recordings. Deleting browser-local storage resets saved device selections.

## Voice models and licenses

VoxWerk uses the [Piper voices catalog](https://huggingface.co/rhasspy/piper-voices). Voice models are distributed separately from the VoxWerk package. Model licenses and attribution requirements vary; review the license and model card linked for each voice before using or redistributing it. The catalog's Hugging Face repository is the source of truth for each model's terms.

## Development

Run the tests from the repository root:

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src
```

Build a wheel:

```sh
python -m pip install build
python -m build --wheel
```

The package layout is `src/voxwerk`; the browser interface is packaged from `src/voxwerk/web/`.

## Project license

No project-level license has been selected yet. Until one is added, the VoxWerk source is not granted an open-source license. This is separate from the licenses for Piper and its individual voice models.