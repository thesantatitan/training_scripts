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
    .run_commands('uv pip install --system --compile-bytecode sentencepiece protobuf datasets')
    .run_function(cache_model)
    .add_local_file('./renders_dataset.jsonl', '/renders_dataset.jsonl',copy=True)
    .run_commands('cd training_scripts && git pull')
)

objaverse_volume = modal.CloudBucketMount(
    bucket_name='objaverse-renders',
    bucket_endpoint_url='https://9a0ea449e510c0a28780f7b8ebb740c8.r2.cloudflarestorage.com',
    secret=modal.Secret.from_name('r2-secret')
)

model_volume = modal.Volume.from_name('models_storage')

@app.function(gpu="H100", image=image, volumes={'/datadisk':objaverse_volume, '/model_storage':model_volume}, timeout=600000)  # defining a Modal Function with a GPU
def start_training():
    import subprocess
    command = [
        "accelerate", "launch", "/training_scripts/flux-control/train_control_lora_flux.py",
        "--pretrained_model_name_or_path=black-forest-labs/FLUX.1-dev",
        "--jsonl_for_train=/renders_dataset.jsonl",
        "--output_dir=/model_storage/flux-control-lora",
        "--mixed_precision=bf16",
        "--train_batch_size=8",
        "--rank=64",
        "--gradient_accumulation_steps=1",
        "--gradient_checkpointing",
        "--learning_rate=1e-5",
        "--report_to=wandb",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        "--max_train_steps=5000",
        "--validation_image=path/to/validation/image.png",
        "--validation_prompt=your validation prompt here",
        "--seed=42",
        "--resolution_widht=2048",
        "--resolution_height=1536",
        "--dataloader_num_workers=15",
        "--offload"
    ]
    subprocess.run(
        command,
        check=True
    )

@app.local_entrypoint()  # defining a CLI entrypoint
def main():
    print("let's try this .remote-ly on Modal...")
    start_training.remote()
