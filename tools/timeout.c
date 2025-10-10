// Simple timeout utility for macOS and Unix-like systems
// Usage: timeout <seconds> <command> [args...]
// Exits with:
//   - child's exit code if it finishes in time
//   - 124 on timeout (after SIGTERM and SIGKILL if needed)
//   - 125 on internal error

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static void die(int code, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(code);
}

static double now_monotonic_sec(void) {
#ifdef CLOCK_MONOTONIC
    {
        struct timespec ts;
        if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
            return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
        }
    }
#endif
    {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
    }
}

static void sleep_ms(int ms) {
    if (ms <= 0) return;
    struct timespec req;
    req.tv_sec = ms / 1000;
    req.tv_nsec = (long)(ms % 1000) * 1000000L;
    while (nanosleep(&req, &req) == -1 && errno == EINTR) {
        // retry
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <seconds> <command> [args...]\n", argv[0]);
        return 125;
    }

    char *end = NULL;
    double timeout_sec = strtod(argv[1], &end);
    if (end == argv[1] || timeout_sec < 0.0) {
        die(125, "invalid timeout value: %s", argv[1]);
    }

    // Optional grace period from env TIMEOUT_KILL_AFTER (seconds), default 5s
    double kill_after_sec = 5.0;
    const char *env = getenv("TIMEOUT_KILL_AFTER");
    if (env && *env) {
        char *e2 = NULL;
        double v = strtod(env, &e2);
        if (e2 != env && v >= 0.0) kill_after_sec = v;
    }

    char **cmd_argv = &argv[2];

    pid_t pid = fork();
    if (pid < 0) die(125, "fork failed: %s", strerror(errno));

    if (pid == 0) {
        // Child: create its own process group so we can kill the whole tree
        if (setpgid(0, 0) != 0) {
            // best effort; ignore
        }
        execvp(cmd_argv[0], cmd_argv);
        fprintf(stderr, "timeout: execvp failed for '%s': %s\n", cmd_argv[0], strerror(errno));
        _exit(127);
    }

    // Parent: monitor child
    if (setpgid(pid, pid) != 0) {
        // if races, ignore
    }

    int status = 0;
    double start = now_monotonic_sec();
    for (;;) {
        pid_t w = waitpid(pid, &status, WNOHANG);
        if (w == pid) {
            // Child exited
            if (WIFEXITED(status)) return WEXITSTATUS(status);
            if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
            return 125; // unexpected
        } else if (w == -1) {
            if (errno == EINTR) continue;
            die(125, "waitpid failed: %s", strerror(errno));
        }

        double elapsed = now_monotonic_sec() - start;
        if (elapsed >= timeout_sec) {
            // Timeout: send SIGTERM to process group
            fprintf(stderr, "timeout: process %d exceeded %.3fs, sending SIGTERM\n", pid, timeout_sec);
            // negative pid => signal the process group
            kill(-pid, SIGTERM);
            // Wait for grace period
            double wait_start = now_monotonic_sec();
            for (;;) {
                pid_t w2 = waitpid(pid, &status, WNOHANG);
                if (w2 == pid) {
                    if (WIFEXITED(status)) return 124; // timed out but exited after TERM
                    if (WIFSIGNALED(status)) return 124;
                    return 124;
                } else if (w2 == -1 && errno != EINTR) {
                    break;
                }
                double waited = now_monotonic_sec() - wait_start;
                if (waited >= kill_after_sec) break;
                sleep_ms(50);
            }
            fprintf(stderr, "timeout: process %d still running, sending SIGKILL\n", pid);
            kill(-pid, SIGKILL);
            // Reap
            for (;;) {
                pid_t w3 = waitpid(pid, &status, 0);
                if (w3 == pid) break;
                if (w3 == -1 && errno != EINTR) break;
            }
            return 124;
        }

        sleep_ms(50);
    }
}
