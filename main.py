# This code is editable! Click the Run button below to execute.

import modal

app = modal.App("example-get-started")  # creating an App
image = modal.Image.debian_slim()

@app.function(gpu="H100:8", image=image)  # defining a Modal Function with a GPU
def check_gpus():
    import subprocess

    print("here's my gpu:")
    try:
        subprocess.run(["nvidia-smi"], check=True)
    except Exception:
        print("no gpu found :(")


@app.local_entrypoint()  # defining a CLI entrypoint
def main():
    print("hello from the .local playground!")
    check_gpus.local()

    print("let's try this .remote-ly on Modal...")
    check_gpus.remote()
