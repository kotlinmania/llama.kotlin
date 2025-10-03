import java.util.Locale

plugins {
    kotlin("multiplatform") version "2.2.20"
    kotlin("plugin.serialization") version "2.2.20"
    id("com.github.ben-manes.versions") version "0.51.0"
    id("maven-publish")
}

group = "ai.solace.llamakotlin"
version = "0.1.0"

repositories {
    mavenCentral()
}

kotlin {
    jvm()
    linuxX64 {
        binaries {
            executable {
                entryPoint = "ai.solace.llamakotlin.main"
            }
            executable("bench") {
                entryPoint = "ai.solace.bench.main"
            }
        }
    }
    macosX64 {
        binaries {
            executable {
                entryPoint = "ai.solace.llamakotlin.main"
            }
            executable("bench") {
                entryPoint = "ai.solace.bench.main"
            }
        }
    }
    macosArm64 {
        binaries {
            executable {
                entryPoint = "ai.solace.llamakotlin.main"
            }
            executable("bench") {
                entryPoint = "ai.solace.bench.main"
            }
        }
    }
    mingwX64 {
        binaries {
            executable {
                entryPoint = "ai.solace.llamakotlin.main"
            }
            executable("bench") {
                entryPoint = "ai.solace.bench.main"
            }
        }
    }

    js(IR) {
        binaries.library()
        browser()
        nodejs()
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation(kotlin("stdlib"))
                implementation("org.jetbrains.kotlinx:atomicfu:0.23.1")
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.2")
                implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.9.0")
                // ZLib.kotlin vendored sources deps
                implementation("org.jetbrains.kotlinx:kotlinx-io-core:0.8.0")
                implementation("co.touchlab:kermit:2.0.8")
                implementation("org.jetbrains.kotlinx:kotlinx-datetime:0.7.1")
            }
            kotlin.srcDir("src/commonMain/kotlin")
            resources.srcDir("src/commonMain/resources")
        }
        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.10.2")
            }
            kotlin.srcDir("src/commonTest/kotlin")
            resources.srcDir("src/commonTest/resources")
        }

        val nativeMain by creating {
            dependsOn(commonMain)
            kotlin.srcDir("src/nativeMain/kotlin")
        }
        val nativeTest by creating {
            dependsOn(commonTest)
            kotlin.srcDir("src/nativeTest/kotlin")
        }

        val linuxX64Main by getting { dependsOn(nativeMain) }
        val linuxX64Test by getting { dependsOn(nativeTest) }
        val macosX64Main by getting { dependsOn(nativeMain) }
        val macosX64Test by getting { dependsOn(nativeTest) }
        val macosArm64Main by getting { dependsOn(nativeMain) }
        val macosArm64Test by getting { dependsOn(nativeTest) }
        val mingwX64Main by getting { dependsOn(nativeMain) }
        val mingwX64Test by getting { dependsOn(nativeTest) }

        val jsMain by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core-js:1.10.2")
            }
        }
        val jsTest by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-test-js:1.10.2")
            }
        }
    }
}

publishing {
    publications {
        withType<MavenPublication> {
            artifactId = "llama.kotlin"
        }
    }
}

// Disassembly helpers for bench executable
val disasmDir = layout.buildDirectory.dir("disasm")

tasks.register<Exec>("disasmBenchMacosArm64") {
    group = "bench"
    description = "Disassemble macOS arm64 bench executable"
    dependsOn("linkBenchReleaseExecutableMacosArm64")
    doFirst {
        disasmDir.get().asFile.mkdirs()
    }
    val exe = layout.buildDirectory.file("bin/macosArm64/benchReleaseExecutable/bench.kexe").get().asFile
    val out = disasmDir.get().file("bench-macosArm64.asm").asFile
    // Prefer llvm-objdump via Xcode toolchain
    commandLine("bash", "-lc", "xcrun llvm-objdump -d '${exe.absolutePath}' > '${out.absolutePath}' || otool -tvV '${exe.absolutePath}' > '${out.absolutePath}'")
}

tasks.register<Exec>("disasmBenchLinuxX64") {
    group = "bench"
    description = "Disassemble Linux x64 bench executable"
    dependsOn("linkBenchReleaseExecutableLinuxX64")
    doFirst {
        disasmDir.get().asFile.mkdirs()
    }
    val exe = layout.buildDirectory.file("bin/linuxX64/benchReleaseExecutable/bench.kexe").get().asFile
    val out = disasmDir.get().file("bench-linuxX64.asm").asFile
    commandLine("bash", "-lc", "objdump -d '${exe.absolutePath}' > '${out.absolutePath}'")
}

// Kotlin-based summaries (avoid shell quoting issues)
abstract class DisasmSummaryTask : DefaultTask() {
    @get:InputFile
    abstract val asmFile: RegularFileProperty

    @get:OutputFile
    abstract val outFile: RegularFileProperty

    @TaskAction
    fun summarize() {
        val txt = asmFile.get().asFile.readText()
        fun count(re: Regex) = re.findAll(txt).count()
        val sb = StringBuilder()
        sb.appendLine("=== Disassembly summary ===")
        sb.appendLine("file: ${asmFile.get().asFile}")
        // Generic counts
        sb.appendLine("vector regs (vN/xmm/ymm): ${count(Regex("\\b(v[0-9]+|xmm[0-9]+|ymm[0-9]+)\\b"))}")
        // arm64 patterns
        sb.appendLine("arm64 ushl: ${count(Regex("\\bushl\\b"))}")
        sb.appendLine("arm64 lsr/ushr: ${count(Regex("\\b(lsr|ushr)\\b"))}")
        sb.appendLine("arm64 lsl/shl: ${count(Regex("\\b(lsl|shl)\\b"))}")
        sb.appendLine("arm64 orr: ${count(Regex("\\borr\\b"))}")
        sb.appendLine("arm64 eor: ${count(Regex("\\beor\\b"))}")
        sb.appendLine("arm64 ext: ${count(Regex("\\bext\\b"))}")
        // x86 patterns
        sb.appendLine("x86 vpsllw/vpsrlw: ${count(Regex("\\bvps(ll|rl)w\\b"))}")
        sb.appendLine("x86 psllw/psrlw: ${count(Regex("\\bps(ll|rl)w\\b"))}")
        sb.appendLine("x86 por/pxor: ${count(Regex("\\bp(x|)or\\b"))}")
        outFile.get().asFile.writeText(sb.toString())
        println(sb.toString())
    }
}

tasks.register<DisasmSummaryTask>("disasmSummaryMacosArm64") {
    group = "bench"
    description = "Summarize vector ops from macOS arm64 disassembly"
    dependsOn("disasmBenchMacosArm64")
    asmFile.set(disasmDir.get().file("bench-macosArm64.asm"))
    outFile.set(disasmDir.get().file("bench-macosArm64.summary.txt"))
}

tasks.register<DisasmSummaryTask>("disasmSummaryLinuxX64") {
    group = "bench"
    description = "Summarize vector ops from Linux x64 disassembly"
    dependsOn("disasmBenchLinuxX64")
    asmFile.set(disasmDir.get().file("bench-linuxX64.asm"))
    outFile.set(disasmDir.get().file("bench-linuxX64.summary.txt"))
}

val hostOs = System.getProperty("os.name")
val hostArch = System.getProperty("os.arch")
val swarBenchTarget = when {
    hostOs == "Mac OS X" && hostArch == "aarch64" -> "macosArm64"
    hostOs == "Mac OS X" -> "macosX64"
    hostOs.startsWith("Linux") -> "linuxX64"
    hostOs.startsWith("Windows") -> "mingwX64"
    else -> null
}

if (swarBenchTarget != null) {
    val targetCapitalized = swarBenchTarget.replaceFirstChar { ch ->
        if (ch.isLowerCase()) ch.titlecase(Locale.ROOT) else ch.toString()
    }
    val benchBinary = layout.buildDirectory.file("bin/$swarBenchTarget/benchReleaseExecutable/bench.kexe")

    tasks.register<Exec>("runSwarBench") {
        group = "bench"
        description = "Run SWAR average benchmark suite on $swarBenchTarget"
        dependsOn("linkBenchReleaseExecutable$targetCapitalized")
        commandLine(benchBinary.get().asFile.absolutePath, "--swar-avg")
    }

    tasks.register<Exec>("runSwarBenchParallel") {
        group = "bench"
        description = "Run parallel SWAR average benchmark suite on $swarBenchTarget"
        dependsOn("linkBenchReleaseExecutable$targetCapitalized")
        commandLine(benchBinary.get().asFile.absolutePath, "--swar-avg-par")
    }

    tasks.register("runAllSwarBenches") {
        group = "bench"
        description = "Run SWAR benchmark suites (serial + parallel) on $swarBenchTarget"
        dependsOn("runSwarBench", "runSwarBenchParallel")
    }
}

// Native test compilation re-enabled for model integration testing
// tasks.configureEach {
//     if (
//         name.contains("Test") && (
//             name.contains("LinuxX64", ignoreCase = true) ||
//             name.contains("MacosX64", ignoreCase = true) ||
//             name.contains("MacosArm64", ignoreCase = true) ||
//             name.contains("MingwX64", ignoreCase = true)
//         )
//     ) {
//         enabled = false
//     }
// }
