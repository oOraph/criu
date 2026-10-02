PLUGIN_NAME := cuda_plugin
PLUGIN_SOBJ := cuda_plugin.so
PLUGIN_SRCS := cuda_plugin.c cuda_plugin_driver.c cuda_plugin_cli.c cuda_wait.c cuda_device_map.c cuda_gpu_pages.c cuda.pb-c.c
PLUGIN_HEADERS := cuda_plugin.h cuda_checkpoint.h cuda_wait.h cuda_device_map.h cuda_gpu_pages.h cuda.pb-c.h
OFFLOAD_BIN := cuda-offload

DEPS_CUDA := $(PLUGIN_SOBJ)

PLUGIN_INCLUDE  	:= -iquote../../include
PLUGIN_INCLUDE  	+= -iquote../../criu/include
PLUGIN_INCLUDE  	+= -iquote../../criu/arch/$(ARCH)/include/
PLUGIN_INCLUDE  	+= -iquote../../

COMPEL := ../../compel/compel-host

PLUGIN_CFLAGS := -g -Wall -Werror -shared -nostartfiles -fPIC

__nmk_dir ?= ../../scripts/nmk/scripts/
include $(__nmk_dir)msg.mk

all: $(DEPS_CUDA) $(OFFLOAD_BIN)

# A multi-target pattern rule runs protoc once for both generated files.
%.pb-c.c %.pb-c.h: %.proto
	$(Q) protoc --proto_path=. --c_out=. $<

# -DBUILD_FOR_CRIU_PLUGIN makes cuda_gpu_pages.c use CRIU's log.h so its pr_*
# output lands in dump.log/restore.log.
cuda_plugin.so: $(PLUGIN_SRCS) $(PLUGIN_HEADERS)
	$(call msg-gen, $@)
	$(Q) $(CC) $(PLUGIN_CFLAGS) $(DEFINES) -DBUILD_FOR_CRIU_PLUGIN $(shell $(COMPEL) includes) $(PLUGIN_SRCS) -o $@ $(PLUGIN_INCLUDE) $(PLUGIN_LDFLAGS) -ldl -lpthread

# Standalone tool: built without -DBUILD_FOR_CRIU_PLUGIN so pr_* goes to stderr.
$(OFFLOAD_BIN): cuda-offload.c cuda_gpu_pages.c cuda_gpu_pages.h
	$(call msg-gen, $@)
	$(Q) $(CC) -g -Wall -Werror -pthread cuda-offload.c cuda_gpu_pages.c -o $@

clean:
	$(call msg-clean, $@)
	$(Q) $(RM) $(PLUGIN_SOBJ) $(OFFLOAD_BIN) cuda.pb-c.c cuda.pb-c.h
.PHONY: clean

mrproper: clean

install:
	$(Q) mkdir -p $(DESTDIR)$(PLUGINDIR)
	$(E) "  INSTALL " $(PLUGIN_NAME)
	$(Q) install -m 755 $(PLUGIN_SOBJ) $(DESTDIR)$(PLUGINDIR)
.PHONY: install

uninstall:
	$(E) " UNINSTALL" $(PLUGIN_NAME)
	$(Q) $(RM) $(DESTDIR)$(PLUGINDIR)/$(PLUGIN_SOBJ)
.PHONY: uninstall
