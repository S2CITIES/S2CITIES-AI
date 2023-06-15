# S2CITIES-AI
Video-based Recognition of the "The Canadian Women's Foundation" Signal for Help

## Installing TensorFlow + MediaPipe on Apple Silicon

1. Install `pip install mediapipe-silicon` which installs `protobuf==3.20.3`.
2. Take the `builder.py` from the installed version and copy it somewhere else `~/.pyenv/versions/s2cities/lib/python3.10/site-packages/google/protobuf/internal`.
3. Install `pip install tensorflow-macos tensorflow-metal` which installs `protobuf==3.19.6`. TensorFlow has priority, so keep this version, but overwrite the `builder.py` with the one you took earlier.

https://stackoverflow.com/questions/71759248/importerror-cannot-import-name-builder-from-google-protobuf-internal