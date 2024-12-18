#install git and other stuff
apt update
apt install -y git
apt install -y build-essentials

#install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/thesantatitan/training_scripts.git
cd training_scripts
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt