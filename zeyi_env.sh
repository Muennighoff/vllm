cd vllm
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python=python3.12 .vllm_venv
source .vllm_venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install --editable .


# # Download and install nvm:
# curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
# \. "$HOME/.nvm/nvm.sh"
# nvm install 22
# node -v
# nvm current
# npm -v
# # Download claude code
# npm install -g @anthropic-ai/claude-code



# nightly env
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python=python3.12 .nightly_venv
source .nightly_venv/bin/activate
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
cd vllm
# uv pip install -e .
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
