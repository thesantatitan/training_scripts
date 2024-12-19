import modal
import torch
import os

hftoken = os.getenv("HF_TOKEN")

app = modal.App("train-flux")  # creating an App

def cache_model():
    from diffusers.pipelines.flux.pipeline_flux_control import FluxControlPipeline
    pipe = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
    
image = (
    modal.Image.from_registry('pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel')
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
    .run_commands('uv pip install --system --compile-bytecode deepspeed')
    .add_local_file('./renders_dataset.jsonl', '/renders_dataset.jsonl',copy=True)
    .add_local_file('./flux-control/train_control_lora_flux.py', '/flux-control/train_control_lora_flux.py',copy=True)
)

objaverse_volume = modal.CloudBucketMount(
    bucket_name='objaverse-renders',
    bucket_endpoint_url='https://9a0ea449e510c0a28780f7b8ebb740c8.r2.cloudflarestorage.com',
    secret=modal.Secret.from_name('r2-secret')
)

model_volume = modal.Volume.from_name('models_storage')

@app.function(gpu="H100:8", image=image, volumes={'/datadisk':objaverse_volume, '/model_storage':model_volume}, timeout=3599)
def check_stuff():
    import subprocess
    modal.interact()
    subprocess.run(["accelerate", "config", '--config_file', '/model_storage/accelerate_config.yaml'], check=True)
    


@app.function(gpu="H100:8", cpu=5, image=image, volumes={'/datadisk':objaverse_volume, '/model_storage':model_volume}, timeout=86400)  # defining a Modal Function with a GPU
def start_training():
    import subprocess
    command = [
        "accelerate", "launch", "--config_file=/model_storage/accelerate_config.yaml", "/flux-control/train_control_lora_flux.py",
        "--pretrained_model_name_or_path=black-forest-labs/FLUX.1-dev",
        "--jsonl_for_train=/renders_dataset.jsonl",
        "--output_dir=/model_storage/flux-control-lora",
        "--mixed_precision=bf16",
        "--train_batch_size=4",
        "--rank=64",
        "--gradient_accumulation_steps=1",
        "--gradient_checkpointing",
        "--learning_rate=1e-4",
        "--report_to=wandb",
        "--lr_scheduler=constant_with_warmup",
        "--lr_warmup_steps=100",
        "--max_train_steps=5000",
        "--seed=42",
        "--resolution_width=1024",
        "--resolution_height=768",
        "--dataloader_num_workers=4",
        "--hub_model_id=thesantatitan/flux-control-orbit",
        "--push_to_hub",
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
    # check_stuff.remote()