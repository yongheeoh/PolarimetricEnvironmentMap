plugins {
    alias(libs.plugins.android.application)
    id("com.chaquo.python") version "17.1.0"
}

chaquopy {
    defaultConfig {
        version = "3.10"
        // Retain this small scientific source package alongside bytecode so each
        // processing run can verify and record the exact bundled source hashes.
        extractPackages("mobile")
        buildPython(providers.environmentVariable("PENVMAP_BUILD_PYTHON").getOrElse(
            "python"))
        pip {
            install("numpy==1.23.3")
            install("scipy==1.8.1")
            install("opencv-python==4.5.1.48")
            install("scikit-image==0.18.3")
        }
    }
}

android {
    namespace = "kr.penvmap.capture"
    compileSdk {
        version = release(37)
    }

    defaultConfig {
        applicationId = "kr.penvmap.capture"
        // LG V30 (the field phone) is Android 9.
        minSdk = 28
        targetSdk = 37
        versionCode = 1
        versionName = "0.1"
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    buildTypes {
        release {
            optimization {
                enable = false
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

dependencies {
    implementation(project(":camera-flir"))
}

// Refresh shared PC kernels automatically before Android consumes Python/assets.
// Only the app's bundled copies change; external calibration sources are read-only.
val processingRepo = rootProject.projectDir.parentFile.parentFile.parentFile
val processingSyncPython = providers.environmentVariable("PENVMAP_SYNC_PYTHON").getOrElse(
    "python")
val syncProcessingSnapshot by tasks.registering(Exec::class) {
    group = "build setup"
    description = "Automatically refresh shared PC processing code and bundled profiles"
    workingDir(processingRepo)
    commandLine(processingSyncPython, "capture/app/tools/sync_processing.py", "--auto")
}

val verifyProcessingSnapshot by tasks.registering(Exec::class) {
    dependsOn(syncProcessingSnapshot)
    group = "verification"
    description = "Verify the bundled Python kernels and calibration snapshots"
    workingDir(processingRepo)
    commandLine(providers.environmentVariable("PENVMAP_BUILD_PYTHON").getOrElse(
        "python"), "capture/app/tools/sync_processing.py", "--check")
}

tasks.matching {
    it.name == "preBuild" || it.name.matches(
        Regex("merge.*PythonSources|generate.*PythonSourceAssets|generate.*PythonProxies|merge.*Assets"))
}.configureEach { dependsOn(verifyProcessingSnapshot) }
