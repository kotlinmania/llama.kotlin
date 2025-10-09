import java.util.Locale
import org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget
import org.jetbrains.kotlin.gradle.tasks.CInteropProcess
import org.jetbrains.kotlin.gradle.tasks.KotlinNativeLink
import org.jetbrains.kotlin.konan.target.KonanTarget
import org.jetbrains.kotlin.gradle.targets.native.tasks.KotlinNativeTest

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

val hostOs = System.getProperty("os.name")
val hostArch = System.getProperty("os.arch")

// ---------------------------------------------------------------------------
// Native coroutine runtimes (kcoro C and C++ variants)
// ---------------------------------------------------------------------------

val kcoroLib = layout.projectDirectory.file("external/kcoro/core/build/lib/libkcoro.a")

val buildKcoro by tasks.registering(Exec::class) {
    group = "kcoro"
    description = "Build kcoro C static library"
    workingDir = file("external/kcoro/core")
    commandLine("make")
    inputs.dir("external/kcoro/core/src")
    inputs.dir("external/kcoro/arch")
    inputs.file("external/kcoro/core/Makefile")
    outputs.file(kcoroLib)
}

val kcoroCppBuildDir = layout.buildDirectory.dir("kcoro_cpp")
val kcoroCppLib = kcoroCppBuildDir.map { it.file("libkcoro_cpp.a") }

val configureKcoroCpp by tasks.registering(Exec::class) {
    group = "kcoro"
    description = "Configure kcoro C++ build"
    commandLine(
        "cmake",
        "-S",
        "external/kcoro_cpp",
        "-B",
        kcoroCppBuildDir.get().asFile.absolutePath,
        "-DCMAKE_BUILD_TYPE=Release"
    )
    inputs.file("external/kcoro_cpp/CMakeLists.txt")
    outputs.file(kcoroCppBuildDir.map { it.file("CMakeCache.txt") })
}

val buildKcoroCpp by tasks.registering(Exec::class) {
    group = "kcoro"
    description = "Build kcoro C++ static library"
    dependsOn(configureKcoroCpp)
    commandLine(
        "cmake",
        "--build",
        kcoroCppBuildDir.get().asFile.absolutePath,
        "--config",
        "Release"
    )
    inputs.dir("external/kcoro_cpp/src")
    inputs.dir("external/kcoro_cpp/arch")
    outputs.file(kcoroCppLib)
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
        compilerOptions {
            target.set("es2015")
            freeCompilerArgs.add("-Xes-long-as-bigint")
        }
    }

    targets.withType<KotlinNativeTarget>().configureEach {
        compilations.getByName("test").cinterops.create("posix")

        if (konanTarget == KonanTarget.MACOS_ARM64) {
            val mainCompilation = compilations["main"]
            val kcoroInclude = layout.projectDirectory.dir("external/kcoro/include")
            val kcoroCppInclude = layout.projectDirectory.dir("external/kcoro_cpp/include")
            val cinteropInclude = layout.projectDirectory.dir("src/nativeInterop/cinterop")
            val kcoroLibDir = layout.projectDirectory.dir("external/kcoro/core/build/lib")
            val kcoroCppLibDirProvider = layout.buildDirectory.dir("kcoro_cpp")

            mainCompilation.cinterops.create("kcoro") {
                definitionFile = file("src/nativeInterop/cinterop/kcoro.def")
                compilerOpts("-I${kcoroInclude.asFile.absolutePath}")
            }

            mainCompilation.cinterops.create("kcoro_cpp") {
                definitionFile = file("src/nativeInterop/cinterop/kcoro_cpp.def")
                compilerOpts(
                    "-I${kcoroCppInclude.asFile.absolutePath}",
                    "-I${cinteropInclude.asFile.absolutePath}"
                )
            }

            binaries.all {
                linkTaskProvider.configure {
                    dependsOn(buildKcoro, buildKcoroCpp)
                }
                linkerOpts(
                    "-L${kcoroLibDir.asFile.absolutePath}",
                    "-L${kcoroCppLibDirProvider.get().asFile.absolutePath}",
                    "-lkcoro",
                    "-lkcoro_cpp",
                    "-lc++"
                )
            }
        }
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
                implementation("org.jetbrains.kotlinx:kotlinx-io-core:0.8.0")
            }
            kotlin.srcDir("src/commonTest/kotlin")
            resources.srcDir("src/commonTest/resources")
        }

        val nativeMain by creating {
            dependsOn(commonMain)
            kotlin.srcDir("src/nativeMain/kotlin")
        }

        val linuxX64Main by getting { dependsOn(nativeMain) }
        val macosX64Main by getting { dependsOn(nativeMain) }
        val macosArm64Main by getting { dependsOn(nativeMain) }
        val mingwX64Main by getting { dependsOn(nativeMain) }

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

tasks.withType<KotlinNativeTest>().configureEach {
    if (name == "macosArm64Test") {
        args = args + "--ktest_logger=SIMPLE"
    }
}

// Kotlin/Native is the primary focus; pause JVM test execution for now.
tasks.named("jvmTest").configure {
    enabled = false
}

tasks.register<Exec>("kcoroBench") {
    group = "bench"
    description = "Run kcoro ping-pong benchmark on macOS arm64."
    onlyIf { hostOs == "Mac OS X" && hostArch == "aarch64" }
    dependsOn("linkDebugTestMacosArm64")
    val testBinary = layout.buildDirectory.file("bin/macosArm64/debugTest/test.kexe")
    doFirst {
        println("[kcoroBench] running PingPongBenchmarkTest on macOS arm64")
    }
    commandLine(
        testBinary.get().asFile.absolutePath,
        "--ktest_filter=ai.solace.bench.PingPongBenchmarkTest.*",
        "--ktest_logger=SIMPLE"
    )
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

tasks.withType<CInteropProcess>().configureEach {
    when {
        name.contains("Kcoro_cpp", ignoreCase = true) -> dependsOn(buildKcoroCpp)
        name.contains("Kcoro", ignoreCase = true) -> dependsOn(buildKcoro)
    }
}

tasks.register<DisasmSummaryTask>("disasmSummaryLinuxX64") {
    group = "bench"
    description = "Summarize vector ops from Linux x64 disassembly"
    dependsOn("disasmBenchLinuxX64")
    asmFile.set(disasmDir.get().file("bench-linuxX64.asm"))
    outFile.set(disasmDir.get().file("bench-linuxX64.summary.txt"))
}

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

    tasks.register<Exec>("runSwarBenchParallel") {
        group = "bench"
        description = "Run parallel SWAR average benchmark suite on $swarBenchTarget"
        dependsOn("linkBenchReleaseExecutable$targetCapitalized")
        commandLine(benchBinary.get().asFile.absolutePath, "--swar-avg-par")
    }

    tasks.register("runAllSwarBenches") {
        group = "bench"
        description = "Run SWAR benchmark suite (parallel) on $swarBenchTarget"
        dependsOn("runSwarBenchParallel")
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
