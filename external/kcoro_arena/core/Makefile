include ../mk/common.mk

AR ?= ar
ARFLAGS ?= rcs

# Public include path
CFLAGS += -I../include

# Always kill lingering kcoro test processes before builds (can disable with KILL_BEFORE=0)
KILL_BEFORE ?= 1
PREKILL := ../scripts/kill_kcoro_procs.sh

OBJDIR := build/obj
BINDIR := build/lib

# C sources (stackless-only, legacy files removed)
# Note: kcoro_core.c removed (contains stackful coroutine code)
# Note: kc_sched.c, kc_scope.c removed (stackful scheduler/scope management)
SRCS := src/kc_chan.c src/kc_actor.c src/kc_cancel.c src/kc_zcopy.c src/kc_runtime_config.c src/kc_bench.c src/kc_dispatch.c src/kc_token_kernel.c src/kc_desc.c src/kc_arena.c src/kc_cpu_monitor.c src/kcoro_stackless.c src/koro_sched_stackless.c src/koro_task.c
# Note: kc_chan_stackless.c temporarily excluded due to incomplete implementation

OBJS := $(patsubst src/%.c,$(OBJDIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

LIB := $(BINDIR)/libkcoro_arena.a

all: prekill $(LIB)

$(BINDIR) $(OBJDIR):
	@mkdir -p $@

$(OBJDIR)/%.o: src/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(LIB): $(BINDIR) $(OBJS)
	$(AR) $(ARFLAGS) $@ $(OBJS)
	@echo "built $@"

clean:
	rm -rf build

.PHONY: prekill kill-tests
prekill:
	@if [ "$(KILL_BEFORE)" = "1" ] && [ -x $(PREKILL) ]; then \
	  bash $(PREKILL); \
	fi

kill-tests:
	@if [ -x $(PREKILL) ]; then bash $(PREKILL); fi

# Verify library freshness: fails if any source newer than built archive
.PHONY: verify-fresh
verify-fresh: $(LIB)
 @latest_src=$(shell find src -type f -name '*.c' -printf '%T@\n' | sort -n | tail -1); \
	 lib_time=$(shell stat -c %Y $(LIB) 2>/dev/null || echo 0); \
	 latest_int=$$(printf '%.0f' $$latest_src); \
	 if [ $$lib_time -lt $$latest_int ]; then \
	   echo "ERROR: libkcoro_arena.a is stale vs sources (lib=$$lib_time < src=$$latest_int)" >&2; exit 1; \
	 else echo "Fresh: lib timestamp $$lib_time >= src $$latest_int"; fi

.PHONY: all clean

-include $(DEPS)
