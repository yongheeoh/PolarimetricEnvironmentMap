# Android capture and processing

FLIR USB capture, MECHA stage control, Gallery and on-device S0/full processing.
Scene groups are `envmap`, `color`, `sun` and `floor`.

## Build

Use JDK 21 and the Android SDK/NDK configured by Gradle. Set `JAVA_HOME` and
`ANDROID_HOME`. If needed, set `PENVMAP_SYNC_PYTHON` to the PC processing Python
and `PENVMAP_BUILD_PYTHON` to Python 3.10 for Chaquopy.

From `capture/app/android`:

```sh
gradlew.bat :capture:assembleDebug
```

Install `capture/build/outputs/apk/debug/capture-debug.apk` on the phone.
The build syncs and verifies shared `pemap` kernels and bundled profiles.
Rebuild and reinstall the APK after source changes.

## Transfer data

From the repository root:

```sh
python capture/app/tools/pull_sessions.py --pull --verify --base <data-directory>
```

Sessions are read from phone `Documents/penvmap`. Basler phone capture is not implemented.
