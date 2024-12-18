set -ex

#install git and other stuff
apt update
apt install -y git
apt install -y build-essential

git clone https://github.com/thesantatitan/training_scripts.git
cd training_scripts
uv venv
ls -la
. .venv/bin/activate
uv pip install -r requirements.txt