RACK_DIR ?= /home/ttuser/code/vcv-cv-harness/RackSDK/Rack-SDK
PYTHON_INC ?= /home/ttuser/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/include/python3.12
PYTHON_LIB ?= /home/ttuser/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/libpython3.12.so
CXXFLAGS += -I$(PYTHON_INC) -fno-gnu-unique
LDFLAGS += $(PYTHON_LIB) -Wl,-rpath,/home/ttuser/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib
include $(RACK_DIR)/plugin.mk
