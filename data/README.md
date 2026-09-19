# Audio regression inputs

Run `.venv/bin/python -m benchmarks.fetch_audio` to obtain the files in `manifest.json`. Original files are cached locally and excluded from Git. SHA-256 hashes identify the exact inputs; decoding and resampling use the locked SoundFile/Torchaudio versions.

- `speech.wav`: VOiCES sample provided in [PyTorch's audio I/O tutorial](https://docs.pytorch.org/audio/2.8/tutorials/audio_io_tutorial.html).
- `environment.wav`: Daniel Simon's steam-train whistle sample provided in [PyTorch's augmentation tutorial](https://docs.pytorch.org/audio/2.6.0/tutorials/audio_data_augmentation_tutorial.html).
- `music.ogg`: **Vibe Ace**, Kevin MacLeod, [Free Music Archive](https://freemusicarchive.org/music/Kevin_MacLeod/Jazz_Sampler/Vibe_Ace), [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). Retrieved from librosa's example-data service and checked against its [official registry](https://github.com/librosa/librosa/blob/main/librosa/util/example_data/registry.txt).

These three recordings provide initial real-audio regressions. They do not represent the diversity of the training data or establish comprehensive model quality.
