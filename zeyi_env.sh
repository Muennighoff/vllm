cd /mnt/ddn/t-zeyichen/vllm
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