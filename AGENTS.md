# Repository Guidelines

## Project Structure & Module Organization

This repository contains a Streamlit fire-detection app and a YOLO-format dataset. `app.py` is the entry point and loads a trained Ultralytics model from `best.pt` by default. `data.yaml` defines the dataset split paths and single `fire` class. Dataset files are organized as `train/`, `valid/`, and `test/`, each with `images/` and `labels/` subfolders. Generated folders such as `runs/`, `__pycache__/`, virtual environments, and local `Ultralytics/` are ignored.

## Build, Test, and Development Commands

- `python -m venv .venv` creates an isolated Python environment.
- `.venv\Scripts\activate` activates the environment on Windows PowerShell.
- `pip install -r requirements.txt` installs Streamlit, Ultralytics, and Pillow.
- `streamlit run app.py` starts the local web app.
- `yolo detect train data=data.yaml model=yolo11n.pt epochs=50 imgsz=640` trains a detector using the included dataset.
- `yolo detect val model=best.pt data=data.yaml imgsz=640` validates a trained model against the configured validation split.

After training, copy `runs/detect/train/weights/best.pt` to the repository root, or set `MODEL_PATH`. Optional runtime settings include `CONFIDENCE_THRESHOLD` and `IMAGE_SIZE`.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation, type hints where they clarify function boundaries, and `pathlib.Path` for filesystem paths. Keep Streamlit UI code readable and top-down. Use uppercase names for constants such as `MODEL_PATH`; use snake_case for functions and variables. Avoid committing caches, training runs, or model checkpoints unless explicitly needed.

## Testing Guidelines

There is no automated test suite in this checkout. Before submitting changes, run `streamlit run app.py` and verify both paths: missing-model guidance and successful inference with a valid `best.pt`. For dataset or training changes, run YOLO validation and inspect precision/recall output. Keep image-label pairs synchronized; every image added under a split should have a matching YOLO `.txt` label file.

## Commit & Pull Request Guidelines

Git history is unavailable in this exported checkout, so use concise imperative commit messages such as `Add model path validation` or `Update dataset config`. Pull requests should describe behavior changed, list commands run, mention dataset or model-file changes, and include screenshots for Streamlit UI updates. Link issues when available and call out required local files, especially `best.pt`.

## Security & Configuration Tips

Do not hard-code absolute local paths, secrets, or machine-specific environment settings. Prefer `MODEL_PATH`, `CONFIDENCE_THRESHOLD`, and `IMAGE_SIZE` for configuration. Treat detection results as advisory only; the app already warns users not to rely on the model as an emergency alert system.
