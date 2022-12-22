# Installare TensorFlow + MediaPipe su Apple Silicon

1. Installo `pip install mediapipe-silicon` che installa `protobuf==3.20.3`
2. Prendo il `builder.py` dalla versione installata da lui e lo copio da qualche parte `~/.pyenv/versions/s2cities/lib/python3.10/site-packages/google/protobuf/internal`
3. Installo `pip install tensorflow-macos tensorflow-metal` che installa `protobuf==3.19.6`, TensorFlow ha la priorità quindi tengo quest'ultima versione, ma sovrascrivo il `builder.py` con quello che ho preso da prima.

https://stackoverflow.com/questions/71759248/importerror-cannot-import-name-builder-from-google-protobuf-internal
