pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}
plugins {
    id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

// penvmap capture app (phone + FLIR + rotation stage). camera-flir is the USB3 Vision driver
// taken from github.com/divisonofficer/filr_android @ 9ee6426; the OAK-D app module is not used.
rootProject.name = "PenvmapCapture"
include(":camera-flir")
include(":capture")
