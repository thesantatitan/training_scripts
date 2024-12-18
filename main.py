import modal
import torch

app = modal.App("example-get-started")  # creating an App
image = modal.Image.debian_slim().add_local_file('./setup.sh', '/setup.sh', copy=True).run_commands('chmod +x /setup.sh && /setup.sh')  # creating an Image with a local file

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
