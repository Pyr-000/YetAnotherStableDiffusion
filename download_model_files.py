from pathlib import Path
try:
  from huggingface_hub import snapshot_download
except ImportError:
  print(f"Unable to import snapshot_download from huggingface_hub! You may need to install the package via 'pip install huggingface_hub'")

from argparse import ArgumentParser
parser = ArgumentParser("SDXL model file acquisition")
parser.add_argument(type=str, nargs="?", default="stabilityai/stable-diffusion-xl-base-1.0", help="model ID on https://huggingface.co", dest="hub_model_id")
parser.add_argument("--name", type=str,  default=None, help="local model name", dest="local_name")
parser.add_argument("--pickle", action="store_true", help="use .bin instead of .safetensors files, as this is the only option with some custom models. USE AT YOUR OWN PERIL! .pt files could contain malicious content.", dest="pt_files")
parser.add_argument("--add-pickle", action="store_true", help="download both .bin and .safetensors files, as this is the only option with some custom models. USE AT YOUR OWN PERIL! .pt files could contain malicious content.", dest="add_pt_files")
parser.add_argument("--full", action="store_true", help="download all files contained within the repository", dest="full")
parser.add_argument("--no-link", action="store_true", help="disable potential usage of symlinks linking to user cache files. Enforce placing actual files in the model dir.", dest="no_link")
args = parser.parse_args()

local_name = args.local_name if args.local_name is not None else args.hub_model_id.split("/",1)[1]
local_name = "".join([char if char.isalnum() or char in [".","-"] else "_" for char in local_name])
while "__" in local_name:
    local_name = local_name.replace("__","_")
local_name = local_name[:64]

target_dir = Path(__file__).parent.joinpath(f"./models/{local_name}")
# crude filter pattern, but correctly covers model files thus far.
dirs = ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "unet", "vae", "scheduler"] # the scheduler dir is really only required for SD2.x v-diffusion models
files = ["*.json", "diffusion_pytorch_model.safetensors", "model.safetensors", "pytorch_model.safetensors", "*.txt"]
if args.add_pt_files:
  files += [x.replace(".safetensors",".bin") for x in files if ".safetensors" in x]
elif args.pt_files:
  files = [x.replace(".safetensors",".bin") for x in files]
if not args.full:
  patterns = ["model_index.json", "LICENSE.md", "README.md"]
  for dir in dirs:
    for file in files:
      patterns.append(f"{dir}/{file}")
else:
  patterns = None

extra_kwargs = {}
if args.no_link:
  extra_kwargs = {"local_dir_use_symlinks":False}

print("Starting download. This should omit any extra data beyond the files relevant to this implementation.")
print("Model weights will be downloaded at full (fp32) precision, suitable for loading in both half (fp16) and full (fp32) mode.")

snapshot_download(args.hub_model_id, local_dir=target_dir, allow_patterns=patterns, **extra_kwargs)

print(f"Saved. Results:")
print(target_dir)
for item in target_dir.glob("*"):
  print(f"    {item.relative_to(target_dir)}")
  if item.is_dir():
    for sub_item in item.glob("*"):
      print(f"        {sub_item.relative_to(item)}")
