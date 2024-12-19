import modal
import pathlib
import shlex
import subprocess
import torch

GRADIO_PORT = 8000
hftoken = os.getenv("HF_TOKEN")
app = modal.App("flux-lora-demo")

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
    .add_local_file('./gradio_app.py', '/gradio_app.py',copy=True)
)

@app.function(
    image=image,
    allow_concurrent_inputs=100,  # Ensure we can handle multiple requests
    concurrency_limit=1,  # Ensure all requests end up on the same container
    gpu='a100',
    timeout=86400
)
@modal.web_server(GRADIO_PORT, startup_timeout=60)
def web_app():
    target = shlex.quote(str('/gradio_app.py'))
    print(target)
    cmd = f"python {target} --host 0.0.0.0 --port {GRADIO_PORT}"
    subprocess.Popen(cmd, shell=True)

# Run with: modal serve modal_app.py
# Or deploy with: modal deploy modal_app.py
