import modal
import torch
import os

hftoken = os.getenv("HF_TOKEN")

app = modal.App("example-get-started")  # creating an App

def cache_model():
    from diffusers.pipelines.flux.pipeline_flux_control import FluxControlPipeline
    pipe = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
    
image = (
    modal.Image.debian_slim()
    .apt_install(["git", "build-essential"])
    .run_commands('git clone https://github.com/thesantatitan/training_scripts.git')
    .run_commands('uv pip install -r training_scripts/requirements.txt --compile-bytecode --system && uv pip install -r training_scripts/flux-control/requirements.txt --compile-bytecode --system')
    .run_commands('uv pip install --system --compile-bytecode huggingface_hub[cli]')
    .run_commands(f'uv pip install --system --compile-bytecode huggingface_hub[hf_transfer]').env({"HF_HUB_ENABLE_HF_TRANSFER":"1"})
    .run_commands(f'huggingface-cli login --token {hftoken}')
    .run_commands(f'uv pip install --system --compile-bytecode wandb').env({"WANDB_API_KEY": os.getenv("WANDB_API_KEY")}).run_commands('wandb login')
    .run_commands('cd training_scripts')
    .run_commands('uv pip install --system --compile-bytecode sentencepiece protobuf')
    .run_function(cache_model)
)

objaverse_volume = modal.CloudBucketMount(
    bucket_name='objaverse-renders',
    bucket_endpoint_url='https://9a0ea449e510c0a28780f7b8ebb740c8.r2.cloudflarestorage.com',
    secret=modal.Secret.from_name('r2-secret')
)


@app.function(gpu="H100", image=image, volumes={'/datadisk':objaverse_volume})  # defining a Modal Function with a GPU
def check_gpus():
    import subprocess

    print("here's my gpu:")
    try:
        subprocess.run(["nvidia-smi"], check=True)
        subprocess.run(['ls', '-l', '/datadisk'], check=True)
    except Exception:
        print("no gpu found :(")
    print(torch.cuda.is_available())


@app.local_entrypoint()  # defining a CLI entrypoint
def main():
    print("hello from the .local playground!")
    check_gpus.local()

    print("let's try this .remote-ly on Modal...")
    check_gpus.remote()
