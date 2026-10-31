plugins {
    alias(libs.plugins.android.library)
}

android {
    namespace = "com.cgjnkim.mobileokad.flir"
    compileSdk {
        version = release(37)
    }

    defaultConfig {
        // Matches :app. MediaStore IS_PENDING, used when writing the GenICam dump.
        minSdk = 28
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

// No native code yet. The USB3 Vision control channel is plain bulk transfers over
// UsbDeviceConnection, so probing the camera needs neither Aravis nor the NDK.
dependencies {
    testImplementation(libs.junit)
}
