import modal
import torch

app = modal.App("example-get-started")  # creating an App
image = (
    modal.Image.debian_slim()
    .apt_install(["git", "build-essential"])
    .run_commands('git clone https://github.com/thesantatitan/training_scripts.git')
    .run_commands('cd training_scripts && uv add -r requirements.txt && uv add -r flux-control/requirements.txt')
)


@app.function(gpu="H100", image=image)  # defining a Modal Function with a GPU
def check_gpus():
    import subprocess

    print("here's my gpu:")
    try:
        subprocess.run(["nvidia-smi"], check=True)
    except Exception:
        print("no gpu found :(")
    torch.cuda.is_available()


@app.local_entrypoint()  # defining a CLI entrypoint
def main():
    print("hello from the .local playground!")
    check_gpus.local()

    print("let's try this .remote-ly on Modal...")
    check_gpus.remote()
