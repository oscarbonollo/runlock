# Makefile
# create venv and install core packages only
VENV_DIR = .venv

# Prefer Python 3.10 on Windows, fall back cleanly
PY = py -3.10
ifeq (, $(shell command -v py 2>/dev/null))
  PY = python3.10
endif
ifeq (, $(shell command -v python3.10 2>/dev/null))
  PY = python
endif

# Core versions (from snapshot)
TORCH_VER = 2.9.0+cu128
TV_VER    = 0.24.0+cu128
TA_VER    = 2.9.0+cu128
TFM_VER   = 4.56.1
NP_VER    = 2.2.6

TORCH_INDEX = https://download.pytorch.org/whl/cu128

VENV_PY  = $(VENV_DIR)/Scripts/python.exe
VENV_PIP = $(VENV_DIR)/Scripts/pip.exe

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
		numpy==$(NP_VER)

.PHONY: clean
clean:
	rm -rf $(VENV_DIR)