# Makefile
# Minimal: create venv + install core dependencies only
# Windows (Git Bash/MSYS2): uses `python`
# Linux (EC2): uses `python3`

VENV_DIR = .venv

# ------------------------------------------------------------
# Select Python interpreter (robust for mingw32-make)
# ------------------------------------------------------------
UNAME_S := $(shell uname -s 2>/dev/null)

# If we're in Git Bash / MSYS2 on Windows, uname contains MINGW or MSYS
ifneq (,$(findstring MINGW,$(UNAME_S)))
  PY = python
else ifneq (,$(findstring MSYS,$(UNAME_S)))
  PY = python
else
  PY = python3
endif

# ------------------------------------------------------------
# Package versions (from your snapshot)
# ------------------------------------------------------------
TORCH_VER = 2.9.0+cu128
TV_VER    = 0.24.0+cu128
TA_VER    = 2.9.0+cu128
TFM_VER   = 4.56.1
NP_VER    = 2.2.6

TORCH_INDEX = https://download.pytorch.org/whl/cu128

# ------------------------------------------------------------
# venv paths (Windows vs Linux)
# ------------------------------------------------------------
VENV_PY_WIN  = $(VENV_DIR)/Scripts/python.exe
VENV_PIP_WIN = $(VENV_DIR)/Scripts/pip.exe

VENV_PY_LNX  = $(VENV_DIR)/bin/python
VENV_PIP_LNX = $(VENV_DIR)/bin/pip

# default to Linux layout
VENV_PY  = $(VENV_PY_LNX)
VENV_PIP = $(VENV_PIP_LNX)

# override to Windows layout if uname indicates Windows shells
ifneq (,$(findstring MINGW,$(UNAME_S)))
  VENV_PY  = $(VENV_PY_WIN)
  VENV_PIP = $(VENV_PIP_WIN)
endif
ifneq (,$(findstring MSYS,$(UNAME_S)))
  VENV_PY  = $(VENV_PY_WIN)
  VENV_PIP = $(VENV_PIP_WIN)
endif

# ------------------------------------------------------------
# Targets
# ------------------------------------------------------------
.PHONY: venv
venv:
	$(PY) -m venv $(VENV_DIR)
	$(VENV_PY) -m pip install -U pip

.PHONY: install
install: venv
	$(VENV_PIP) install --index-url $(TORCH_INDEX) \
		torch==$(TORCH_VER) \
		torchvision==$(TV_VER) \
		torchaudio==$(TA_VER)
	$(VENV_PIP) install \
		transformers==$(TFM_VER) \
		numpy==$(NP_VER) \
		accelerate

.PHONY: clean
clean:
	rm -rf $(VENV_DIR)

.PHONY: help
help:
	@echo "Usage:"
	@echo "  make PY=python install        (Windows / Conda)"
	@echo "  make PY=python3.10 install    (Ubuntu 22.04)"
	@echo "  make PY=/path/to/python install"