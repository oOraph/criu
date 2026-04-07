PLUGIN_NAME := cuda_plugin
PLUGIN_SOBJ := cuda_plugin.so
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

# cuda_gpu_pages.o is shared between the plugin and cuda-offload.
# Compiled with -fPIC so it can be linked into the shared library.
cuda_gpu_pages.o: cuda_gpu_pages.c cuda_gpu_pages.h
	$(call msg-gen, $@)
	$(Q) $(CC) -g -Wall -Werror -fPIC $(PLUGIN_INCLUDE) -c $< -o $@

cuda_plugin.so: cuda_plugin.c cuda_gpu_pages.o
	$(call msg-gen, $@)
	$(Q) $(CC) $(PLUGIN_CFLAGS) $(shell $(COMPEL) includes) $^ -o $@ $(PLUGIN_INCLUDE) $(PLUGIN_LDFLAGS) -lpthread

cuda-offload: cuda-offload.c cuda_gpu_pages.o
	$(call msg-gen, $@)
	$(Q) $(CC) -Wall -Werror -g $^ -o $@ -lpthread

clean:
	$(call msg-clean, $@)
	$(Q) $(RM) $(PLUGIN_SOBJ) $(OFFLOAD_BIN) cuda_gpu_pages.o
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
