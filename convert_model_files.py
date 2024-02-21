from pathlib import Path
from traceback import format_exc
from argparse import ArgumentParser
parser = ArgumentParser("SD -> diffusers model file conversion")
parser.add_argument(type=str, help="path to model/model filename", dest="source_file")
parser.add_argument("--name", type=str,  default=None, help="local model name (output)", dest="local_name")
parser.add_argument("--xl", action="store_true", help="load model file from StableDiffusionXL. Should fail over to SDXL if loading model as a normal SD model fails.", dest="xl")
args = parser.parse_args()

source_file = Path(args.source_file)
if not source_file.exists() and source_file.is_file():
  print(f"{source_file} does not resolve to a valid file.")
  exit(-1)

local_name = args.local_name if args.local_name is not None else source_file.stem
local_name = "".join([char if char.isalnum() or char in [".","-"] else "_" for char in local_name])
while "__" in local_name:
    local_name = local_name.replace("__","_")
local_name = local_name[:64]

target_dir = Path(__file__).parent.joinpath(f"./models/{local_name}")

from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionXLPipeline
if not args.xl:
  pipeline_class = StableDiffusionPipeline
else:
  pipeline_class = StableDiffusionXLPipeline

# args.source_file as this function does not work with Path() objects
try:
  pipe = pipeline_class.from_single_file(args.source_file)
except Exception:
  # try it with the other pipeline class
  exc = format_exc()
  try:
    print(f"failed to load as {'SDXL'if args.xl else 'base SD'} - trying {'SDXL'if not args.xl else 'base SD'}")
    pipeline_class = StableDiffusionPipeline if args.xl else StableDiffusionXLPipeline
    pipe = pipeline_class.from_single_file(args.source_file)
  except Exception:
    exc2 = format_exc()
    print(f"Failed to load both as standard SD and SDXL!")
    print(f"Standard SD:")
    print(exc)
    print(f"SDXL:")
    print(exc2)
    exit(-1)

pipe.save_pretrained(target_dir)

print(f"Saved. Results:")
print(target_dir)
for item in target_dir.glob("*"):
  print(f"    {item.relative_to(target_dir)}")
  if item.is_dir():
    for sub_item in item.glob("*"):
      print(f"        {sub_item.relative_to(item)}")
