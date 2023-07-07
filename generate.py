import gc
import os
# before transformers/diffusers are imported
os.environ["TRANSFORMERS_VERBOSITY"] = "error" #if not verbose else "info"
os.environ["DIFFUSERS_VERBOSITY"] = "error" #if not verbose else "info"
import base64
import math
import random
from typing import Tuple
import torch
# from torch import autocast
from transformers import models as transformers_models
from diffusers import models as diffusers_models
from transformers import CLIPTextModel, CLIPTokenizer, AutoFeatureExtractor, CLIPSegProcessor, CLIPSegForImageSegmentation
from diffusers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, IPNDMScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler, HeunDiscreteScheduler, DEISMultistepScheduler, UniPCMultistepScheduler
from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel
from diffusers.utils import is_accelerate_available
import argparse
from datetime import datetime
from PIL.PngImagePlugin import PngInfo
import codecs
import cv2
import numpy as np
from traceback import print_exc
from PIL import Image, ImageOps, ImageEnhance, ImageColor
from tqdm.auto import tqdm
from time import time
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
import inspect
from io import BytesIO
from pathlib import Path
import re
from tokens import get_huggingface_token
import uuid
from dataclasses import dataclass

# for automated downloads via huggingface hub
model_id = "CompVis/stable-diffusion-v1-4"
# for manual model installs
models_local_dir = "models/stable-diffusion-2-1"
# for metadata, written during model load
using_local_unet, using_local_vae = False,False
# location of textual inversion concepts (if present). (recursively) search for any .bin files. see see: https://huggingface.co/sd-concepts-library
concepts_dir = "models/concepts"
# one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, ort, mps, xla, lazy, vulkan, meta, hpu
# devices will be set by argparser if used.
DIFFUSION_DEVICE = "cuda"
IO_DEVICE = "cuda"
# execution device when offloading enabled. Required as a global for now.
OFFLOAD_EXEC_DEVICE = "cuda"
# display current image in img2img cycle mode via cv2.
IMAGE_CYCLE_DISPLAY_CV2 = True
# display preprocessed image via cv2 when utilising controlnet preprocessing
PREPROCESS_DISPLAY_CV2 = True
# Window title for cv2 display
CV2_TITLE="Output"
# Window title for cv2 display when displaying preprocessing image
PREPROCESS_CV2_TITLE="ControlNet Input"
# global flag signifying if a v_prediction model is in use.
V_PREDICTION_MODEL = False
# will be read and overridden from the text encoder layer count on model load. Inclusive range limit. 12 for ViT-L/14 (SD1.x) or 23 for ViT-H (SD2.x)
MAX_CLIP_SKIP = 12
# when mixing prompts via concatenation, this will drop embeddings with a weight of zero, instead of adding them as "empty" (zeroed) tensors. Makes a (slight) difference when interpolating prompts, especially for longer sequences.
CONCAT_DROP_ZERO_WEIGHT_EMBEDDINGS = True
# when interpolating prompts, skew ramp-up/ramp-down of prompts, so that both values cross over at the set point. Increasing from 0.5 should reduce lack of definition at the crossover point in concat mode (dynamic length), but will compress or reduce variety in-between prompts.
# should be between 0 and 1, with values >=0.5 being recommended. 0.5 corresponds to no skew.
PROMPT_INTERPOLATION_CROSSOVER_POINT = 0.5
# pixels of separation between individual images in an image grid.
GRID_IMAGE_SEPARATION = 1
# background color of grid images. Accepts any color definition understood by 'PIL.ImageColor', including hex (#rrggbb) and common HTML color names ("darkslategray"). Will be visible both between images (separation) and in empty grid spots (e.g. 8 images on a 3x3 grid)
GRID_IMAGE_BACKGROUND_COLOR = "#2A2B2E"
assert ImageColor.getrgb(GRID_IMAGE_BACKGROUND_COLOR) # check requested color for validity before running the script.
# the non-refined rd64 model variant produces less precise results, but this may yield slightly more reliable behavior with the current thresholding method
SAFETY_PROCESSOR_CLIPSEG_MODEL = "CIDAS/clipseg-rd64-refined" # "CIDAS/clipseg-rd64" # "CIDAS/clipseg-rd16"
# debug option, displays blur heatmaps via cv2 when the safety processor applies masked blur.
SAFETY_PROCESSOR_DISPLAY_MASK_CV2 = False
# Can be used to manually disable xformers memory efficient attention.
XFORMERS_ENABLED = True

OUTPUT_DIR_BASE = "outputs"
OUTPUTS_DIR = f"{OUTPUT_DIR_BASE}/generated"
ANIMATIONS_DIR = f"{OUTPUT_DIR_BASE}/animate"
INDIVIDUAL_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "individual")
UNPUB_DIR = os.path.join(OUTPUTS_DIR, "unpub")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(INDIVIDUAL_OUTPUTS_DIR, exist_ok=True)
os.makedirs(UNPUB_DIR, exist_ok=True)
os.makedirs(ANIMATIONS_DIR, exist_ok=True)

IMPLEMENTED_SCHEDULERS = ["lms", "pndm", "ddim", "euler", "euler_ancestral", "mdpms", "sdpms", "kdpm2", "kdpm2_ancestral", "heun", "deis", "unipc"] # ipndm not supported
V_PREDICTION_SCHEDULERS = IMPLEMENTED_SCHEDULERS # ["euler", "ddim", "mdpms", "euler_ancestral", "pndm", "lms", "sdpms"]
IMPLEMENTED_GS_SCHEDULES = [None, "sin", "cos", "isin", "icos", "fsin", "anneal5", "ianneal5", "rand", "frand"]
# preprocessors for controlnet input images
AUX_PREPROCESSORS = ["detect_pose", "detect_mlsd", "detect_hed"] # , "detect_midas"] # seems to not be in pypi yet, hold until next release
IMPLEMENTED_CONTROLNET_PREPROCESSORS = ["canny"] + AUX_PREPROCESSORS
# short names for specifying controlnet models. will translate requests for name into requests for CONTROLNET_SHORTNAMES[name]
CONTROLNET_SHORTNAMES = {name:f"lllyasviel/sd-controlnet-{name}" for name in ["canny","depth","hed","mlsd","normal","openpose","scribble","seg"]} | {name:f"thibaud/controlnet-{name}-diffusers" for name in [f"sd21-{x}" for x in ["canny","depth","hed","openpose","scribble","zoedepth","color"]]}

# sd2.0 default negative prompt
DEFAULT_NEGATIVE_PROMPT = "" # e.g. dreambot SD2.0 default : "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft"

# THIS WILL AUTOMATICALLY RESET. DO NOT ADD CUSTOM VALUES HERE. substitutions/shortcuts in prompts. Will be filled with substitions for multi-token custom embeddings.
PROMPT_SUBST = {}
# Custom combinations of "shortname":"substitution string", could be added here.
CUSTOM_PROMPT_SUBST = {
    "<negative>":"ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft",
}

# active LoRA, if any. Stored globally for easy access when writing metadata on outputs
ACTIVE_LORA = None
ACTIVE_LORA_WEIGHT = 1.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(type=str, nargs="?", default=None, help="text prompt for generation. Leave unset to enter prompt loop mode. Multiple prompts can be separated by ||.", dest="prompt")
    parser.add_argument("-ii", "--init-img", type=str, default=None, help="use img2img mode. path to the input image", dest="init_img_path")
    parser.add_argument("-st", "--strength", type=float, default=0.75, help="strength of initial image in img2img. 0.0 -> no new information, 1.0 -> only new information", dest="strength")
    parser.add_argument("-s","--steps", type=int, default=50, help="number of sampling steps", dest="steps")
    parser.add_argument("-sc","--scheduler", type=str, default="mdpms", choices=IMPLEMENTED_SCHEDULERS, help="scheduler used when sampling in the diffusion loop", dest="sched_name")
    parser.add_argument("-e", "--ddim-eta", type=float, default=0.0, help="eta adds additional random noise when sampling, only applied on ddim sampler. 0 -> deterministic", dest="ddim_eta")
    parser.add_argument("-es","--ddim-eta-seed", type=int, default=None, help="secondary seed for the eta noise with ddim sampler and eta>0", dest="eta_seed")
    parser.add_argument("-H","--H", type=int, default=512, help="image height, in pixel space", dest="height")
    parser.add_argument("-W","--W", type=int, default=512, help="image width, in pixel space", dest="width")
    parser.add_argument("-n", "--n-samples", type=int, default=1, help="how many samples to produce for each given prompt. A.k.a. batch size", dest="n_samples")
    parser.add_argument("-seq", "--sequential_samples", action="store_true", help="Run batch in sequence instead of in parallel. Removes VRAM requirement for increased batch sizes, increases processing time.", dest="sequential_samples")
    parser.add_argument("-cs", "--scale", type=float, default=9, help="(classifier free) guidance scale (higher values may increse adherence to prompt but decrease 'creativity')", dest="guidance_scale")
    parser.add_argument("-S","--seed", type=int, default=None, help="initial noise seed for reproducing/modifying outputs (None will select a random seed)", dest="seed")
    parser.add_argument("--unet-full", action='store_false', help="Run diffusion UNET at full precision (fp32). Default is half precision (fp16). Increases memory load.", dest="half")
    parser.add_argument("--latents-half", action='store_true', help="Generate half precision latents (fp16). Default is full precision latents (fp32), memory usage will only reduce by <1MB. Outputs will be slightly different.", dest="half_latents")
    parser.add_argument("--diff-device", type=str, default="cuda", help="Device for running diffusion process", dest="diffusion_device")
    parser.add_argument("--io-device", type=str, default="cuda", help="Device for running text encoding and VAE decoding. Keep on CPU for reduced VRAM load.", dest="io_device")
    parser.add_argument("--animate", action="store_true", help="save animation of generation process. Very slow unless --io_device is set to \"cuda\"", dest="animate")
    parser.add_argument("-in", "--interpolate", nargs=2, type=str, help="Two image paths for generating an interpolation animation", default=None, dest='interpolation_targets')
    parser.add_argument("-inx", "--interpolate_extend", type=float, help="Interpolate beyond the specified images in the latent space by the given factor. Disabled with 0. Unlikely to provide useful results.", default=0, dest='interpolation_extend')
    parser.add_argument("--no-check-nsfw", action="store_true", help="Disable NSFW check entirely, slightly speeding up generation.", dest="no_check")
    parser.add_argument("-spl", "--safety-processing-level", type=int, default=3, help="Set level of content safety processing. 7/6 functions like the diffusers pipeline default, 5/4/3 applies a local blur, 2/1 does not process images outside of special labels.", dest="safety_processing_level")
    parser.add_argument("-pf", "--prompts-file", type=str, help="Path of file containing prompts. One line per prompt.", default=None, dest='prompts_file')
    parser.add_argument("-ic", "--image-cycles", type=int, help="Repetition count when using image2image. Will interpolate between multiple prompts when || is used.", default=0, dest='img_cycles')
    parser.add_argument("-cni", "--cycle-no-save-individual", action="store_false", help="Disables saving of individual images in image cycle mode.", dest="image_cycle_save_individual")
    parser.add_argument("-csi", "--cycle-select-image", action="store_true", help="When cycling with batch sizes >1, manually select one of the outputs as the next input image ('guided evolution')", dest="image_cycle_select")
    parser.add_argument("-iz", "--image-zoom", type=int, help="Amount of zoom (pixels cropped per side) for each subsequent img2img cycle", default=0, dest='img_zoom')
    parser.add_argument("-ir", "--image-rotate", type=int, help="Amount of rotation (counter-clockwise degrees) for each subsequent img2img cycle", default=0, dest='img_rotation')
    parser.add_argument("-it", "--image-translate", type=int, help="Amount of translation (x,y tuple in pixels) for each subsequent img2img cycle", default=None, nargs=2, dest='img_translation')
    parser.add_argument("-irc", "--image-rotation-center", type=int, help="Center of rotational axis when applying rotation (0,0 -> top left corner) Default is the image center", default=None, nargs=2, dest='img_center')
    parser.add_argument("-ics", "--image-cycle-sharpen", type=float, default=1.0, help="Sharpening factor applied when zooming and/or rotating. Reduces effect of constant image blurring due to resampling. Sharpens at >1.0, softens at <1.0", dest="cycle_sharpen")
    parser.add_argument("-icc", "--image-color-correction", action="store_true", help="When cycling images, keep lightness and colorization distribution the same as it was initially (LAB histogram cdf matching). Prevents 'magenta shifting' with multiple cycles.", dest="cycle_color_correction")
    parser.add_argument("-cfi", "--cycle-fresh-image", action="store_true", help="When cycling images (-ic), create a fresh image (use text-to-image) in each cycle. Useful for interpolating prompts (especially with fixed seed)", dest="cycle_fresh")
    parser.add_argument("-cb", "--cuda-benchmark", action="store_true", help="Perform CUDA benchmark. Should improve throughput when computing on CUDA, but may slightly increase VRAM usage.", dest="cuda_benchmark")
    parser.add_argument("-as", "--attention-slice", type=int, default=None, help="Set UNET attention slicing slice size. 0 for recommended head_count//2, 1 for maximum memory savings", dest="attention_slicing")
    parser.add_argument("-co", "--cpu-offload", action='store_true', help="Set to enable CPU offloading through accelerate. This should enable compatibility with minimal VRAM at the cost of speed.", dest="cpu_offloading")
    parser.add_argument("-gsc","--gs-schedule", type=str, default=None, choices=IMPLEMENTED_GS_SCHEDULES, help="Set a schedule for variable guidance scale. Default (None) corresponds to no schedule.", dest="gs_schedule")
    parser.add_argument("-om","--online-model", type=str, default=None, help="Set an online model id for acquisition from huggingface hub.", dest="online_model")
    parser.add_argument("-lm","--local-model", type=str, default=None, help="Set a path to a directory containing local model files (should contain unet and vae dirs, see local install in readme).", dest="local_model")
    parser.add_argument("-od","--output-dir", type=str, default=None, help="Set an override for the base output directory. The directory will be created if not already present.", dest="output_dir")
    parser.add_argument("-mn","--mix-negative-prompts", action="store_true", help="Mix negative prompts directly into the prompt itself, instead of using them as uncond embeddings.", dest="negative_prompt_mixing")
    parser.add_argument("-dnp","--default-negative-prompt", type=str, default="", help="Specify a default negative prompt to use when none are specified.", dest="default_negative")
    parser.add_argument("-cls", "--clip-layer-skip", type=int, default=0, help="Skip last n layers of CLIP text encoder. Reduces processing/interpretation level of prompt. Recommended for some custom models", dest="clip_skip_layers")
    parser.add_argument("-rnc", "--re-encode", type=str, default=None, help="Re-encode an image or folder using the VAE of the loaded model. Works using latents saved in PNG metadata", dest="re_encode")
    parser.add_argument("-sel", "--static-embedding-length", type=int, default=None, help="Process prompts without dynamic length. A value of 77 should reproduce results of previous/other implementations. Switches from embedding concatenation to mixing.", dest="static_length")
    parser.add_argument("-mc", "--mix_concat", action="store_true", help="When mixing prompts, perform a concatenation of their embeddings instead of calculating a weighted sum.", dest="mix_mode_concat")
    parser.add_argument("-lop", "--lora-path", type=str, default=None, nargs="+", help="Path(s) to LoRA embedding(s). Will be loaded via diffusers attn_procs / lora_diffusion / lora converter.", dest="lora_path")
    parser.add_argument("-low", "--lora-weight", type=float, default=[0.8], nargs="+", help="Weight(s) of LoRA embedding(s) loaded via --lora-path. Will wrap around if there are less weights than embeddings.", dest="lora_weight")
    parser.add_argument("-com","--controlnet-model", type=str, default=None, help=f"Set a name, path or hub id pointing to a controlnet model. Known names: {CONTROLNET_SHORTNAMES}", dest="controlnet_model")
    parser.add_argument("-coi","--controlnet-input", type=str, default=None, help="Set a path pointing to an input image for the controlnet.", dest="controlnet_input")
    parser.add_argument("-cost", "--controlnet-strength", type=float, default=1.0, help="strength (cond scale) for applying a controlnet", dest="controlnet_strength")
    parser.add_argument("-cosc","--controlnet-schedule", type=str, default=None, choices=IMPLEMENTED_GS_SCHEDULES, help="Set a schedule for controlnet strength. Default (None) corresponds to no schedule.", dest="controlnet_schedule")
    parser.add_argument("-cop","--controlnet-preprocessor", type=str, default=None, choices=IMPLEMENTED_CONTROLNET_PREPROCESSORS, help="Set a preprocessor for controlnet inputs.", dest="controlnet_preprocessor")
    parser.add_argument("-gr", "--guidance-rescale", type=float, default=0.66, help="Set the guidance rescale factor 'phi' to improve image exposure distribution. Disable with a value of 0. Authors of arxiv:2305.08891 recommend values between 0.5 and 0.75", dest="guidance_rescale")
    parser.add_argument("-spr","--second-pass-resize", type=float, default=1, help="Perform a second pass with image size multiplied by this factor. Initially sampling at reduced size can improve consistency and speed with large resolutions. Enabled for values >1.", dest="second_pass_resize")
    parser.add_argument("-sps","--second-pass-steps", type=int, default=50, help="Number of sampling steps at which the new size is used when starting_resize is specified.", dest="second_pass_steps")
    parser.add_argument("-spc","--second-pass-controlnet", action="store_true", help="Use the first pass image in the selected controlnet for the second pass (instead of using img2img). For batch sizes >1, only the first image from the first pass will be used (outside of sequential mode).", dest="second_pass_use_controlnet")
    return parser.parse_args()

def main():
    global DIFFUSION_DEVICE, IO_DEVICE
    global model_id, models_local_dir
    global OUTPUT_DIR_BASE, OUTPUTS_DIR, INDIVIDUAL_OUTPUTS_DIR, UNPUB_DIR, ANIMATIONS_DIR
    global DEFAULT_NEGATIVE_PROMPT
    global ACTIVE_LORA, ACTIVE_LORA_WEIGHT
    args = parse_args()
    if args.cpu_offloading:
        args.diffusion_device, args.io_device = OFFLOAD_EXEC_DEVICE, OFFLOAD_EXEC_DEVICE
    DIFFUSION_DEVICE = args.diffusion_device
    IO_DEVICE = args.io_device
    DEFAULT_NEGATIVE_PROMPT = args.default_negative
    ACTIVE_LORA = args.lora_path
    ACTIVE_LORA_WEIGHT = args.lora_weight

    if args.online_model is not None:
        # set the online model id
        model_id = args.online_model
        # set local model to None to disable local override
        models_local_dir = None
    if args.local_model is not None:
        # local model dirs function as an override - online id can remain untouched
        models_local_dir = args.local_model

    if args.output_dir is not None:
        OUTPUT_DIR_BASE = args.output_dir
        OUTPUTS_DIR = f"{OUTPUT_DIR_BASE}/generated"
        ANIMATIONS_DIR = f"{OUTPUT_DIR_BASE}/animate"
        INDIVIDUAL_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "individual")
        UNPUB_DIR = os.path.join(OUTPUTS_DIR, "unpub")
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        os.makedirs(INDIVIDUAL_OUTPUTS_DIR, exist_ok=True)
        os.makedirs(UNPUB_DIR, exist_ok=True)
        os.makedirs(ANIMATIONS_DIR, exist_ok=True)

    if args.cuda_benchmark:
        torch.backends.cudnn.benchmark = True

    # load up models
    tokenizer, text_encoder, unet, vae, rate_nsfw = load_models(half_precision=args.half, cpu_offloading=args.cpu_offloading)
    if args.no_check:
        rate_nsfw = lambda x: False
    # create generate function using the loaded models

    # if interpolation is requested, run interpolation instead!
    if args.interpolation_targets is not None:
        # swap places between UNET and VAE to speed up large batch decode.
        unet = unet.to(IO_DEVICE)
        vae = vae.to(DIFFUSION_DEVICE)
        create_interpolation(args.interpolation_targets[0], args.interpolation_targets[1], args.steps, vae, args.interpolation_extend)
        exit()

    if args.re_encode is not None:
        unet = unet.to(IO_DEVICE)
        vae = vae.to(DIFFUSION_DEVICE)
        enc_path = Path(args.re_encode)
        if not enc_path.exists():
            print(f"Selected path does not exist: {enc_path}")
            exit()
        if enc_path.is_file():
            targets = [enc_path]
        elif enc_path.is_dir():
            targets = [f for f in enc_path.rglob("*.png")]
        print(f"Accumulated {len(targets)} target files for re-encode!")
        for item in tqdm(targets, desc="Re-encoding"):
            try:
                encoded = re_encode(item, vae)
                meta = load_metadata_from_image(item)
                meta["filename"] = str(item)
                meta["re-encode"] = f"{models_local_dir} (local)" if using_local_vae else model_id
                save_output(p=None, imgs=encoded, argdict=None, data_override=meta)
            except KeyboardInterrupt:
                exit()
            except:
                print_exc()
        exit()

    controlnet_model = load_controlnet(args.controlnet_model,DIFFUSION_DEVICE,args.cpu_offloading)

    generator = QuickGenerator(tokenizer,text_encoder,unet,vae,IO_DEVICE,DIFFUSION_DEVICE,rate_nsfw,args.half_latents)
    generator.configure(
        args.n_samples, args.width, args.height, args.steps, args.guidance_scale, args.seed, args.sched_name, args.ddim_eta, args.eta_seed, args.strength,
        args.animate, args.sequential_samples, True, True, args.attention_slicing, True, args.gs_schedule,
        args.negative_prompt_mixing, args.clip_skip_layers, args.static_length, args.mix_mode_concat,
        controlnet_model, args.controlnet_strength, args.controlnet_preprocessor, args.controlnet_schedule,
        args.safety_processing_level, args.guidance_rescale, args.second_pass_resize, args.second_pass_steps, args.second_pass_use_controlnet
    )

    init_image = None if args.init_img_path is None else Image.open(args.init_img_path).convert("RGB")
    controlnet_input = None if args.controlnet_input is None else Image.open(args.controlnet_input).convert("RGB")

    if args.img_cycles > 0:
        generator.perform_image_cycling(
            args.prompt, args.img_cycles, args.image_cycle_save_individual, True, init_image, args.cycle_fresh, args.cycle_color_correction, None,
            args.img_zoom, args.img_rotation, args.img_center, args.img_translation, args.cycle_sharpen,
            controlnet_input=controlnet_input, select_next_image_in_batch=args.image_cycle_select
        )
        exit()

    if args.prompts_file is not None:
        lines = [line.strip() for line in codecs.open(args.prompts_file, "r").readlines()]
        for line in tqdm(lines, position=0, desc="Input prompts"):
            generator.one_generation(line, init_image, controlnet_input=controlnet_input)
    elif args.prompt is not None:
        generator.one_generation(args.prompt, init_image, controlnet_input=controlnet_input)
    else:
        while True:
            prompt = input("Prompt> ").strip()
            try:
                generator.one_generation(prompt, init_image, controlnet_input=controlnet_input)
            except KeyboardInterrupt:
                pass
            except Exception as e:
                print_exc()

def recurse_set_xformers(item, max_recursion_depth=16):
    if not XFORMERS_ENABLED:
        return
    if hasattr(item, "set_use_memory_efficient_attention_xformers"):
        item.set_use_memory_efficient_attention_xformers(True,None)
    recurse_targets = item.__dict__.values() if hasattr(item, "__dict__") else item.values() if isinstance(item, dict) else item if isinstance(item, list) or hasattr(item, "__iter__") else None
    if max_recursion_depth > 0 and recurse_targets is not None:
        for x in recurse_targets:
            recurse_set_xformers(x,max_recursion_depth-1)

def load_models(half_precision=False, unet_only=False, cpu_offloading=False, vae_only=False, lora_override:str=None, lora_weight_override:float=None):
    global using_local_unet, using_local_vae, PROMPT_SUBST, MAX_CLIP_SKIP, ACTIVE_LORA, ACTIVE_LORA_WEIGHT
    # re-set substitution list (re-loading model would otherwise re-add custom embedding substitutions)
    PROMPT_SUBST = CUSTOM_PROMPT_SUBST
    if cpu_offloading:
        if not is_accelerate_available():
            print("accelerate library is not installed! Unable to utilise CPU offloading!")
            cpu_offloading=False
        else:
            # NOTE: this only offloads parameters, but not buffers by default.
            from accelerate import cpu_offload

    torch.no_grad() # We don't need gradients where we're going.
    tokenizer:transformers_models.clip.tokenization_clip.CLIPTokenizer
    text_encoder:transformers_models.clip.modeling_clip.CLIPTextModel
    unet:diffusers_models.unet_2d_condition.UNet2DConditionModel
    vae:diffusers_models.AutoencoderKL

    unet_model_id = model_id
    vae_model_id = model_id
    text_encoder_id = "openai/clip-vit-large-patch14" # sd 1.x default
    tokenizer_id = "openai/clip-vit-large-patch14" # sd 1.x default
    use_auth_token_unet=get_huggingface_token()
    use_auth_token_vae=get_huggingface_token()

    default_local_model_files_per_dir = [["config.json"], ["diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors"]]
    def dir_has_model_files(dirpath, files_groups=None):
        files_groups = files_groups if files_groups is not None else default_local_model_files_per_dir
        return all([any([os.path.exists(os.path.join(dirpath, f)) for f in group]) for group in files_groups])
    if models_local_dir is not None and len(models_local_dir) > 0:
        unet_dir = os.path.join(models_local_dir, "unet")
        if dir_has_model_files(unet_dir):
            use_auth_token_unet=False
            print(f"Using local unet model files! ({models_local_dir})")
            unet_model_id = unet_dir
            using_local_unet = True
        else:
            print(f"Using unet '{unet_model_id}' (no local data present at {unet_dir})")
        vae_dir = os.path.join(models_local_dir, "vae")
        if dir_has_model_files(vae_dir):
            use_auth_token_vae=False
            print(f"Using local vae model files! ({models_local_dir})")
            vae_model_id = vae_dir
            using_local_vae = True
        else:
            print(f"Using vae '{vae_model_id}' (no local data present at {vae_dir})")
        text_encoder_dir = os.path.join(models_local_dir, "text_encoder")
        if dir_has_model_files(text_encoder_dir, [["config.json"], ["pytorch_model.bin", "pytorch_model.safetensors"]]):
            print(f"Using local text encoder! ({models_local_dir})")
            text_encoder_id = text_encoder_dir
        else:
            print(f"Using text encoder '{text_encoder_id}' (no local data present at {text_encoder_dir})")
        tokenizer_dir = os.path.join(models_local_dir, "tokenizer")
        if all([os.path.exists(os.path.join(tokenizer_dir, f)) for f in ["merges.txt","vocab.json"]]):
            print(f"Using local tokenizer! ({models_local_dir})")
            tokenizer_id = tokenizer_dir
        else:
            print(f"Using tokenizer '{tokenizer_id}' (no local data present at {tokenizer_dir})")

        # crude check to find out if a v_prediction model is present. To be replaced by proper "model default config" at some point.
        global V_PREDICTION_MODEL
        scheduler_config_file_path = os.path.join(models_local_dir, "scheduler/scheduler_config.json")
        if os.path.exists(scheduler_config_file_path):
            with open(scheduler_config_file_path) as f:
                V_PREDICTION_MODEL = '"v_prediction"' in f.read()
        else:
            V_PREDICTION_MODEL = False

        ACTIVE_LORA = lora_override if lora_override is not None else ACTIVE_LORA
        ACTIVE_LORA_WEIGHT = lora_weight_override if lora_weight_override is not None else ACTIVE_LORA_WEIGHT if ACTIVE_LORA_WEIGHT is not None else [1.0]
        ACTIVE_LORA_WEIGHT = [ACTIVE_LORA_WEIGHT] if isinstance(ACTIVE_LORA_WEIGHT, float) else ACTIVE_LORA_WEIGHT
        # to disable LoRA via override, pass "". No override keeps the current LoRA setting to keep re-loading functionality unchanged
        if ACTIVE_LORA not in [None, "", []]:
            try:
                if isinstance(ACTIVE_LORA, str):
                    ACTIVE_LORA = ACTIVE_LORA.split(";")
                if isinstance(ACTIVE_LORA_WEIGHT, str):
                    ACTIVE_LORA_WEIGHT = [try_float(x,1.0) for x in ACTIVE_LORA_WEIGHT.split(";")]
                if len(ACTIVE_LORA_WEIGHT) < len(ACTIVE_LORA): # if there are less weights than embeddings, extend them to the correct length by wrapping around to the first value
                    ACTIVE_LORA_WEIGHT *= int(len(ACTIVE_LORA)/len(ACTIVE_LORA_WEIGHT))+1
                    ACTIVE_LORA_WEIGHT = ACTIVE_LORA_WEIGHT[:len(ACTIVE_LORA)]
                # used to pass one 'pipe' object to lora_diffusion.patch_pipe: expects 'pipe' object with properties pipe.unet, pipe.text_encoder, pipe.tokenizer
                pseudo_pipe_container = lambda **kwargs: type("PseudoPipe", (), kwargs)
                active_lora_paths = []
                active_lora_weights = []
                for lora_item, lora_weight in zip(ACTIVE_LORA, ACTIVE_LORA_WEIGHT):
                    if not (Path(lora_item).exists() and Path(lora_item).is_file()):
                        if Path(lora_item+".safetensors").exists() and Path(lora_item+".safetensors").is_file():
                            lora_item += ".safetensors"
                        elif Path(lora_item+".pt").exists() and Path(lora_item+".pt").is_file():
                            lora_item += ".pt"
                        elif Path(lora_item+".bin").exists() and Path(lora_item+".bin").is_file():
                            lora_item += ".bin"
                        else:
                            tqdm.write(str(ValueError(f"LoRA path {lora_item} could not be parsed to a valid file.")))
                            continue
                    if not (lora_item.endswith(".pt") or lora_item.endswith(".safetensors") or lora_item.endswith(".bin")):
                        tqdm.write(str(ValueError(f"LoRA file {lora_item} should have either '.pt'/'.bin' or '.safetensors' extension, specifying the file format. Assuming .pt for non-safetensors files.")))
                        continue
                    active_lora_paths.append(lora_item)
                    active_lora_weights.append(lora_weight)
                ACTIVE_LORA = active_lora_paths
                ACTIVE_LORA_WEIGHT = active_lora_weights
                def patch_wrapper(unet, text_encoder, tokenizer):
                    global ACTIVE_LORA, ACTIVE_LORA_WEIGHT
                    loaded_lora = []
                    loaded_lora_weights = []
                    for lora, weight in zip(ACTIVE_LORA, ACTIVE_LORA_WEIGHT):
                        try:
                            if patch_worker(unet, text_encoder, tokenizer, lora, weight):
                                loaded_lora.append(lora)
                                loaded_lora_weights.append(weight)
                        except Exception as e:
                            tqdm.write(f"lora {lora} could not be applied: {e}")
                    ACTIVE_LORA = loaded_lora
                    ACTIVE_LORA_WEIGHT = loaded_lora_weights
                def patch_worker(unet, text_encoder, tokenizer, lora_item, item_weight:1.0):
                    try:
                        path_item = Path(lora_item)
                        if path_item.suffix == ".safetensors":
                            from safetensors.torch import load_file #, safe_open
                            load_data = load_file(path_item)
                            """with safe_open(path_item, framework="pt") as handle:
                                if handle.metadata() is not None:
                                    print(f"metadata for {lora_item}:")
                                    pprint(handle.metadata())"""
                        else:
                            load_data = torch.load(path_item)
                    except ImportError as e:
                        tqdm.write(f"Could not load LoRa due to missing library: {e}")
                        return False
                    except ValueError as e:
                        print_exc()
                        tqdm.write(f"Could not load LoRa file: {e}")
                        return False

                    encoder_is_sd1 = text_encoder.get_input_embeddings().weight.data.shape[1] == 768

                    # first, try loading as a diffusers attn_procs (most specific format)
                    try:
                        sd1_len_keys = [k for k in load_data.keys() if len(getattr(load_data[k], "shape", [])) >=2 and 768 == load_data[k].shape[1]]
                        is_sd1 = all([("transformer_blocks.0.attn2.processor.to_" in x and "_lora.down.weight" in x) for x in sd1_len_keys]) and len(sd1_len_keys) >= 32
                        if is_sd1 != encoder_is_sd1: # crude version check, maybe replace with something better
                            raise ValueError(f"{lora_item} contains weights for {'SD2.x' if encoder_is_sd1 else 'SD1.x'}, but encoder is {'SD1.x' if encoder_is_sd1 else 'SD2.x'}")
                        unet.load_attn_procs(load_data)
                        tqdm.write(f"{lora_item} loaded as a diffusers attn_proc!")
                        return True
                    except Exception as e:
                        # print_exc()
                        pass
                    #tqdm.write(f"{lora_item} could not be loaded as diffusers attn_procs")

                    # second, try loading as lora_diffusers
                    try:
                        if not ["<s1>"] in load_data.keys():
                            raise ValueError(f"embedding is not lora_diffusers.")
                        elif load_data["<s1>"].shape[0] == 768 != encoder_is_sd1: # crude version check, maybe replace with something better
                            raise ValueError(f"{lora_item} contains weights for {'SD2.x' if encoder_is_sd1 else 'SD1.x'}, but encoder is {'SD1.x' if encoder_is_sd1 else 'SD2.x'}")
                        from lora_diffusion import patch_pipe, tune_lora_scale
                        patch_only_unet = text_encoder is None or tokenizer is None
                        pseudo_pipe = pseudo_pipe_container(unet=unet,text_encoder=text_encoder,tokenizer=tokenizer)
                        patch_pipe(pseudo_pipe, lora_item, patch_text=not patch_only_unet, patch_ti=not patch_only_unet)
                        tune_lora_scale(unet, item_weight)
                        if not patch_only_unet:
                            tune_lora_scale(text_encoder, item_weight)
                        tqdm.write(f"{lora_item} loaded via lora_diffusers!")
                        return True
                    except ImportError:
                        tqdm.write("Unable to load lora_diffusion. Please install via 'pip install git+https://github.com/cloneofsimo/lora.git' to use lora_diffusion embeddings.")
                        pass
                    except Exception as e:
                        # print_exc()
                        pass
                    #tqdm.write(f"{lora_item} could not be loaded via lora_diffusion.")

                    # finally, try to use lora converter to load
                    try:
                        sd1_len_keys = [k for k in load_data.keys() if len(getattr(load_data[k], "shape", [])) >=2 and 768 == load_data[k].shape[1] and "lora_down.weight" in k]
                        if (len(sd1_len_keys) >= 92) != encoder_is_sd1: # crude version check, maybe replace with something better
                            print(f"{lora_item} contains weights for {'SD2.x' if encoder_is_sd1 else 'SD1.x'}, but encoder is {'SD1.x' if encoder_is_sd1 else 'SD2.x'}")
                            return False
                        load_lora_convert(load_data,unet=unet, text_encoder=text_encoder, merge_strength=item_weight)
                        tqdm.write(f"{lora_item} loaded via LoRA converter!")
                        return True
                    except Exception as e:
                        #print_exc()
                        pass
                    #tqdm.write(f"{lora_item} could not be loaded via LoRa converter.")

                    tqdm.write(f"{lora_item} embedding failed to load using known methods: diffusers attn_proc / lora_diffusion / lora converter")
                    return False

            except ValueError as e:
                print(e)
                ACTIVE_LORA = None
                def patch_wrapper(*args, **kwargs):
                    pass
        else:
            ACTIVE_LORA = None
            def patch_wrapper(*args, **kwargs):
                pass

    if not vae_only:
        # Load the UNet model for generating the latents.
        if half_precision:
            unet = UNet2DConditionModel.from_pretrained(unet_model_id, subfolder="unet", use_auth_token=use_auth_token_unet, torch_dtype=torch.float16, revision="fp16")
        else:
            unet = UNet2DConditionModel.from_pretrained(unet_model_id, subfolder="unet", use_auth_token=use_auth_token_unet)

        if cpu_offloading:
            cpu_offload(unet, execution_device=OFFLOAD_EXEC_DEVICE, offload_buffers=True)

        if unet_only:
            patch_wrapper(unet,None,None)
            recurse_set_xformers(unet)
            return unet

    # Load the autoencoder model which will be used to decode the latents into image space.
    if False:
        vae = AutoencoderKL.from_pretrained(vae_model_id, subfolder="vae", use_auth_token=use_auth_token_vae, torch_dtype=torch.float16)
    else:
        vae = AutoencoderKL.from_pretrained(vae_model_id, subfolder="vae", use_auth_token=use_auth_token_vae)
    if vae_only:
        recurse_set_xformers(vae)
        return vae

    # Load the tokenizer and text encoder to tokenize and encode the text.
    #tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    #text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_id)
    text_encoder = CLIPTextModel.from_pretrained(text_encoder_id)

    try:
        MAX_CLIP_SKIP = len(text_encoder.text_model.encoder.layers)
    except Exception:
        print_exc()
        print("Unable to infer encoder hidden layer count (CLIP skip limit) from the text encoder!")
    print(f"Encoder hidden layer count (max CLIP skip): {MAX_CLIP_SKIP}")

    #rate_nsfw = get_safety_checker(cpu_offloading=cpu_offloading)
    rate_nsfw = SafetyChecker(cpu_offloading=False)
    concepts_path = Path(concepts_dir)
    available_concepts = [f for f in concepts_path.rglob("*.bin")] + [f for f in concepts_path.rglob("*.pt")]
    if len(available_concepts) > 0:
        print(f"Adding {len(available_concepts)} Textual Inversion concepts found in {concepts_path}: ")
        for item in available_concepts:
            try:
                load_learned_embed_in_clip(item, text_encoder, tokenizer)
            except Exception as e:
                print(f"Loading concept from {item} failed: {e}")
        print("")

    patch_wrapper(unet,text_encoder,tokenizer)

    # text encoder should be offloaded after adding custom embeddings to ensure that every embedding is actually on the same device.
    if cpu_offloading:
        cpu_offload(vae, execution_device=OFFLOAD_EXEC_DEVICE, offload_buffers=True)
        cpu_offload(text_encoder, execution_device=OFFLOAD_EXEC_DEVICE, offload_buffers=True)

    recurse_set_xformers([tokenizer, text_encoder, unet, vae, rate_nsfw])
    return tokenizer, text_encoder, unet, vae, rate_nsfw

# take state dict, apply LoRA to unet and text encoder (at strength)
def load_lora_convert(state_dict, unet=None, text_encoder=None, merge_strength=0.75):
    # adapted from https://github.com/haofanwang/diffusers/blob/75501a37157da4968291a7929bb8cb374eb57f22/scripts/convert_lora_safetensor_to_diffusers.py
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    LORA_PREFIX_UNET = "lora_unet"
    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key and LORA_PREFIX_TEXT_ENCODER in key:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = text_encoder
            if text_encoder is None:
                continue
        elif LORA_PREFIX_UNET in key:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = unet
            if unet is None:
                continue
        else:
            print(f"skipping unknown LoRA key {key}") # -> {state_dict[key]}")
            continue

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if "conv" in key:
            lora_conv_up = torch.nn.Conv2d(curr_layer.in_channels, state_dict[pair_keys[0]].shape[1], curr_layer.kernel_size, curr_layer.stride, curr_layer.padding, bias=False)
            lora_conv_down = torch.nn.Conv2d(state_dict[pair_keys[1]].shape[0], curr_layer.out_channels, 1, 1, bias=False)
            lora_conv_up.weight.data = state_dict[pair_keys[0]]
            lora_conv_down.weight.data = state_dict[pair_keys[1]]
            setattr(curr_layer,"lora_conv_up", lora_conv_up)
            setattr(curr_layer,"lora_conv_down", lora_conv_down)
            # apply default conv forward, then add lora conv forward results, at merge strength
            curr_layer:torch.nn.Conv2d
            def hook(module, input, output):
                # input will be a tuple of "only the positional arguments given to the module", which should only be the input tensor itself.
                if isinstance(input, tuple) and len(input) == 1:
                    input = input[0]
                lora_dtype_down = module.lora_conv_up.weight.data.dtype
                lora_dtype_up = module.lora_conv_down.weight.data.dtype
                return (output.to(lora_dtype_up) + module.lora_conv_up(module.lora_conv_down(input.to(lora_dtype_down)).to(lora_dtype_up)) * merge_strength).to(input.dtype)
            curr_layer.register_forward_hook(hook)

        elif len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += merge_strength * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += merge_strength * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)
        #except Exception as e:
            # print(f"LoRA converter {e} | {key} | {curr_layer.weight.data.shape} | {state_dict[pair_keys[0]].shape} | {state_dict[pair_keys[1]].shape}")

def try_int(s:str, default:int=None):
    try:
        return int(s)
    except Exception:
        return default
def try_float(s:str, default:float=None):
    try:
        return float(s)
    except Exception:
        return default

def load_controlnet(source:str,device="cpu",cpu_offloading=False):
    if source in [None, ""]:
        return None
    source = CONTROLNET_SHORTNAMES.get(source, source)
    controlnet = None
    try:
        controlnet = ControlNetModel.from_pretrained(source, torch_dtype=torch.float16).to(device)
        if cpu_offloading:
            if not is_accelerate_available():
                print("accelerate library is not installed! Unable to utilise CPU offloading!")
                cpu_offloading=False
            else:
                # NOTE: this only offloads parameters, but not buffers by default.
                from accelerate import cpu_offload
                cpu_offload(controlnet, execution_device=OFFLOAD_EXEC_DEVICE, offload_buffers=True)
        recurse_set_xformers(controlnet)
        setattr(controlnet, "_MODEL_NAME", source)
    except Exception as e:
        print_exc()
        print(f"failed to load controlnet: {e}")
    return controlnet

def controlnet_preprocess(image:Image.Image, width:int, height:int, preprocessor:str=None):
    image = image.resize((width,height), resample=Image.Resampling.LANCZOS)
    image = image.convert("RGB")
    if preprocessor == "canny":
        image = Image.fromarray(cv2.Canny(np.array(image),100,200)).convert("RGB")
    elif preprocessor in AUX_PREPROCESSORS:
        processed = controlnet_aux_extractor(preprocessor,image)
        image = image if processed is None else processed
        # controlnet_aux only accepts one value for target resolution. For now, just resize in post.
        image = image.resize((width,height), resample=Image.Resampling.LANCZOS).convert("RGB")
    processed_PIL = image
    if PREPROCESS_DISPLAY_CV2:
        show_image(image, PREPROCESS_CV2_TITLE)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image, processed_PIL

def generate_segmented(
        tokenizer:transformers_models.clip.tokenization_clip.CLIPTokenizer,
        text_encoder:transformers_models.clip.modeling_clip.CLIPTextModel,
        unet:diffusers_models.unet_2d_condition.UNet2DConditionModel,
        vae:diffusers_models.AutoencoderKL,
        IO_DEVICE="cpu",UNET_DEVICE="cuda",rate_nsfw=(lambda x: False),half_precision_latents=False
    ):
    if not vae.device == torch.device("meta"):
        vae.to(IO_DEVICE)
        vae_compute_device = IO_DEVICE
    else:
        vae_compute_device = OFFLOAD_EXEC_DEVICE
    if not text_encoder.device == torch.device("meta"):
        text_encoder.to(IO_DEVICE)
    if not unet.device == torch.device("meta"):
        unet.to(UNET_DEVICE)
        unet_compute_device = UNET_DEVICE
    else:
        unet_compute_device = OFFLOAD_EXEC_DEVICE

    # xformers attention will fail to function on CPU.
    unet.set_use_memory_efficient_attention_xformers(unet_compute_device=="cuda" and XFORMERS_ENABLED,None)
    vae.set_use_memory_efficient_attention_xformers(vae_compute_device=="cuda" and XFORMERS_ENABLED,None)

    UNET_DEVICE = UNET_DEVICE if UNET_DEVICE != "meta" else OFFLOAD_EXEC_DEVICE
    IO_DEVICE = IO_DEVICE if IO_DEVICE != "meta" else OFFLOAD_EXEC_DEVICE
    #if UNET_DEVICE == "meta" and unet.dtype == torch.float16:
    #    half_precision_latents = True

    def process_prompt_segmentation(prompt, default_skip=0):
        segment_weights = []
        processed_segments = []
        segment_skips = []
        # the start token is not part of the segments - add a leading weight of 1 and a leading skip of <default> for it.
        token_weights = [1]
        token_skips = [default_skip]
        # split out segments where custom weights are defined. Use non-greedy content matching to identify the segment: '.+?'
        weighted_segments = re.split("\((.+?:-?[\.0-9;]+)\)", prompt)
        for segment in weighted_segments:
            segment = segment.strip()
            # empty segments will exclusively show up with zero-length prompts, which produce one empty segment. Their handling can be ignored.
            segment_weight = 1
            segment_skip = default_skip
            split_by_segment_weight = re.split(":(-?[\.0-9;]+)$",segment)
            if len(split_by_segment_weight)>1:
                # should result in _exactly_ three segments: prompt, weight capture group, trailing empty segment (caused by the '\)$' at the end of the regex expression
                # multiple captures shall be impossible, as the regex requires the end of the string '$'
                assert len(split_by_segment_weight) == 3
                # take first item of regex split (the prompt). leading '(' will have been removed by the split, as it is outside the capture group.
                segment = split_by_segment_weight[0].strip()
                # take second item (captured prompt weight with potential skip) - leading ':' and trailing ')' shall have been excluded from the capture
                weight_item = split_by_segment_weight[1].strip()
                # split off a potential skip value
                weight_item_split = weight_item.split(";")
                if len(weight_item_split) > 2:
                    tqdm.write(f"NOTE: malformed prompt segment value: {weight_item}")
                segment_weight = try_float(weight_item_split[0],1)
                segment_skip = default_skip if len(weight_item_split)<=1 else try_int(weight_item_split[1],default_skip)
            processed_segments.append(segment)
            segment_weights.append(segment_weight)
            segment_skips.append(segment_skip)
            segment_tokens = tokenizer(segment,padding=False,max_length=None,truncation=False,return_tensors="pt",return_overflowing_tokens=False).input_ids[0]
            # tokenize segment without padding, remove leading <|startoftext|> and trailing <|endoftext|> to measure the segment token count. see note below about max_length=None
            segment_token_count = len(segment_tokens)-2
            token_weights += [segment_weight]*segment_token_count
            token_skips += [segment_skip]*segment_token_count

        # join segments back together, but keep a space between then (otherwise words would amalgamate at the join points)
        prompt = " ".join(processed_segments)
        return prompt, token_weights, token_skips

    # build attention io_data in reduced form
    def weight_summary_item_string(i, prev_seg_count, prev_weight, prev_skip, text_input):
        segment_start_index = i - prev_seg_count
        # from start_index to (inclusive) start_index+count-1 - ':start_index+count' as the second index of l[a:b] is exclusive
        segment_text = tokenizer.batch_decode([text_input.input_ids[0][segment_start_index:segment_start_index+prev_seg_count]])[0].strip()
        segment_text = segment_text if not len(segment_text) > 14 else segment_text[:5]+"..."+segment_text[-5:]
        return f" ({prev_weight}{';'+str(prev_skip) if prev_skip>0 else ''}@{segment_text})"
    def prompt_weight_summary_string(token_weights, token_skips, text_input):
        summary = ""
        if True: #len(token_weights) > 2: # do not run processing on empty prompts.
            prev_seg=(token_weights[1],token_skips[1])
            prev_seg_count = 1
            for i,seg in enumerate(zip(token_weights,token_skips)):
                if i <= 1:
                    # skip value of leading start token. Additionally, skip first weight, as it is initialized outside the loop. not excluded from the list to keep enum indices correct.
                    continue
                if prev_seg == seg:
                    prev_seg_count += 1
                else:
                    # extract first token of attention value start - first index is <count> before the current index (which is no longer included), +1 as the weights start after startoftext
                    summary+=(weight_summary_item_string(i, prev_seg_count, prev_seg[0], prev_seg[1], text_input))
                    prev_seg = seg
                    prev_seg_count = 1
            # append final weight and count, which would occur from the subsequent weight i+1
            summary+=(weight_summary_item_string(i+1, prev_seg_count, prev_seg[0], prev_seg[1], text_input))
        else:
            summary = ""
        return summary.strip()

    def encode_embedding_vectors(text_chunks, attention_mask, token_skips, default_clip_skip):
        if all([skip <= 0 for skip in token_skips]): # a value of 0 (index: [-1]) would yield the final layer output, equivalent to default behavior
            # unpack state, batch dim from every encode
            processed_chunks = [text_encoder(text_chunk.to(IO_DEVICE), attention_mask=attention_mask)[0][0] for text_chunk in text_chunks]
            token_counts = [len(c) for c in processed_chunks]
            # re-attach chunks
            embedding_vectors = torch.cat(processed_chunks)
            return embedding_vectors, token_counts
        else:
            # [text_encoder.text_model.final_layer_norm(text_encoder(text_chunk.to(IO_DEVICE), attention_mask=attention_mask, output_hidden_states=True).hidden_states[-(clip_skip_layers+1)]) for text_chunk in text_chunks]
            # encode every chunk, retrieve all hidden layers
            text_chunks_hidden_states = [text_encoder(text_chunk.to(IO_DEVICE), attention_mask=attention_mask, output_hidden_states=True).hidden_states for text_chunk in text_chunks]
            # embeddings must be available for any token skip level requested, as well as for the default skip level (applied on start/end/padding tokens)
            embedding_vectors = {
                required_skip_level : torch.cat([text_encoder.text_model.final_layer_norm(chunk_hidden_states[-(required_skip_level+1)])[0] for chunk_hidden_states in text_chunks_hidden_states])
                for required_skip_level in set(token_skips+[default_clip_skip])
            }
            # use the default skip level to extract per-chunk token counts
            processed_chunks_of_default_skip = [text_encoder.text_model.final_layer_norm(chunk_hidden_states[-(default_clip_skip+1)])[0] for chunk_hidden_states in text_chunks_hidden_states]
            token_counts = [len(c) for c in processed_chunks_of_default_skip]
            # re-build resulting embedding vectors by interleaving the embedding vectors for different skip levels according to token_skips_after_start
            new_embedding_vectors = []
            # iterate over the vector count of the embedding (equal length for every decoded skip level)
            for i in range(len(embedding_vectors[default_clip_skip])):
                # retrieve the skip value of state i, use the default value if reading states of trailing end/padding tokens without a specified value.
                skip_level = default_clip_skip if not i in range(len(token_skips)) else token_skips[i]
                # select the embedding of the requested skip level, pick the vector at the index of the current iteration
                new_embedding_vectors.append(embedding_vectors[skip_level][i])
            embedding_vectors = torch.stack(new_embedding_vectors)
            return embedding_vectors, token_counts

    def apply_embedding_vector_weights(embedding_vectors, token_weights):
        # Ignore (ending and padding) vectors outside of the stored range of token weights. Their weight will be 1.
        return torch.stack([emb_vector if not i in range(len(token_weights)) else emb_vector*token_weights[i] for i,emb_vector in enumerate(embedding_vectors)])

    @torch.no_grad()
    def perform_text_encode(prompt,clip_skip_layers=0,pad_to_length=None,prefer_default_length=False):
        io_data = {}

        prompt = prompt.strip()
        clip_skip_prompt_override = re.search('\{cls[0-9]+\}$', prompt) # match only '{cls<int>}' at the end of the prompt
        if clip_skip_prompt_override is not None:
            clip_skip_layers = min(int(clip_skip_prompt_override[0][4:-1]), MAX_CLIP_SKIP) # take match, drop leading '{cls' and trailing '}'. Clamp to possible value limit
            # since this only matches on the end of the string, splitting multiple times on unfortunate prompts is not possible
            prompt = re.split('\{cls[0-9]+\}$', prompt)[0]
        io_data["clip_skip_layers"] = [clip_skip_layers]
        # set rerun layers *after* checking for skip override. this allows producing of 'special tensors' with a trailing clip skip override
        rerun_self_kwargs = {"clip_skip_layers":clip_skip_layers,"pad_to_length":pad_to_length,"prefer_default_length":prefer_default_length}

        # apply all prompt substitutions. Process longest substitution first, to avoid substring collisions.
        sorted_substitutions = sorted([(k,PROMPT_SUBST[k],uuid.uuid4()) for k in PROMPT_SUBST], key=lambda x: len(x[0]), reverse=True)
        # first, grab all substitutions and replace them by a temporary UUID. This avoids substring collisions between substitution values and subsequent (shorter) substitution keys.
        for (p,subst,identifier) in sorted_substitutions:
            prompt = prompt.replace(p, f"<{identifier}>")
        # then, replace UUID placeholders with substitution values. Substring collisions should not be an issue.
        for (p,subst,identifier) in sorted_substitutions:
            prompt = prompt.replace(f"<{identifier}>", subst)

        # text embeddings
        prompt, token_weights, token_skips = process_prompt_segmentation(prompt,clip_skip_layers)

        text_input = tokenizer(
            prompt,
            padding="max_length" if pad_to_length is not None else False,
            max_length=pad_to_length, # NOTE: None may be treated as "auto-select length limit of model" according to docs - however, no limit seems to be set with both L/14 (SD1.x) and OpenClip (SD2.x)
            truncation=(pad_to_length is not None),
            return_tensors="pt",
            return_overflowing_tokens=(pad_to_length is not None),
        )
        # if the prompt is shorter than the standard token limit without a specified padding setting, but default-size embeddings are preferred, rerun with default pad length.
        if len(text_input.input_ids[0]) <= tokenizer.model_max_length and pad_to_length is None and prefer_default_length:
            return perform_text_encode(prompt,pad_to_length=tokenizer.model_max_length,prefer_default_length=False,**{k:v for k,v in rerun_self_kwargs.items() if not k in ["pad_to_length","prefer_default_length"]})
        num_truncated_tokens = getattr(text_input,"num_truncated_tokens",[0])
        io_data["text_readback"] = [t.strip() for t in tokenizer.batch_decode(text_input.input_ids, cleanup_tokenization_spaces=False)]
        # SD2.0 openCLIP seems to pad with '!'. compact these down.
        trailing_char_counts = []
        for i, readback_string in enumerate(io_data["text_readback"]):
            new_count = 0
            for char in readback_string[::-1]: # reverse string -> move inwards from the back
                if char == '!':
                    new_count += 1
                else:
                    break
            if new_count > 0: # [:-0] would be [:0], thus eliminating the entire string.
                io_data["text_readback"][i] = readback_string[:-new_count]
            trailing_char_counts.append(new_count)
        # if a batch element had items truncated, the remaining token count is the negative of the amount of truncated tokens
        # otherwise, count the amount of <|endoftext|> in the readback. Sidenote: this will report incorrectly if the prompt is below the token limit and happens to contain "<|endoftext|>" for some unthinkable reason.
        io_data["remaining_token_count"] = [(- item) if item>0 else (io_data["text_readback"][idx].count("<|endoftext|>") - 1) if item!=0 and pad_to_length is not None else float('inf') for idx,item in enumerate(num_truncated_tokens)]
        # If there is more space when looking at SD2.0 padding (trailing '!'), output it instead.
        io_data["remaining_token_count"] = [max(remaining, trailing) for (remaining,trailing) in zip(io_data["remaining_token_count"], trailing_char_counts)]
        io_data["text_readback"] = [s.replace("<|endoftext|>","").replace("<|startoftext|>","") for s in io_data["text_readback"]]
        #io_data["attention"] = text_input.attention_mask.cpu().numpy().tolist() # unused, as the text encoders of both SD1.x and SD2.x lack a "use_attention_mask" attribute denoting manual attention maps.
        try:
            io_data["weights"] = [prompt_weight_summary_string(token_weights, token_skips, text_input)]
        except:
            io_data["weights"] = ""

        chunk_count = (len(text_input.input_ids[0]) // tokenizer.model_max_length) + (1 if (len(text_input.input_ids[0])%tokenizer.model_max_length) > 0 else 0)
        text_chunks = torch.chunk(text_input.input_ids[0], chunk_count)
        # reassemble chunk items into batches of size 1
        text_chunks = [torch.stack([chunk]) for chunk in text_chunks]

        if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
            attention_mask = text_input.attention_mask.to(IO_DEVICE)
        else:
            attention_mask = None

        embedding_vectors, io_data["tokens"] = encode_embedding_vectors(text_chunks, attention_mask, token_skips, clip_skip_layers)

        embedding_vectors = apply_embedding_vector_weights(embedding_vectors, token_weights)
        # repack batch dimension around the embedding vectors
        text_embeddings = torch.stack([embedding_vectors])

        special_embeddings = get_special_embeddings(prompt.strip(), rerun_self_kwargs)
        if special_embeddings is not None:
            text_embeddings = special_embeddings

        return text_embeddings, io_data

    def get_special_embeddings(prompt, run_encode_kwargs):
        if prompt == "<damaged_uncond>":
            uncond, _ = perform_text_encode("", **run_encode_kwargs)
            for i in range(uncond.shape[1]):
                # create some tensor damage: mix half of the vector elements with a random element of the same vector.
                idx = torch.randperm(uncond.shape[2])
                for j in range(uncond.shape[2]):
                    if torch.rand(1) > 0.5:
                        uncond[:,i,j] = (uncond[:,i,j]*2 + uncond[:,i,idx[j]]) /3
            return uncond
        elif prompt == "<alpha_dropout_uncond>":
            uncond, _ = perform_text_encode("", **run_encode_kwargs)
            return torch.nn.functional.alpha_dropout(uncond, p=0.5)
        elif prompt == "<all_starts>":
            uncond, _ = perform_text_encode("", **run_encode_kwargs)
            for i in range(1,uncond.shape[1]):
                uncond[:,i] = uncond[:,0]
            return uncond
        elif prompt == "<all_ends>":
            uncond, _ = perform_text_encode("", **run_encode_kwargs)
            for i in range(0,uncond.shape[1]-1):
                uncond[:,i] = uncond[:,-1]
            return uncond
        elif prompt == "<all_start_end>":
            uncond, _ = perform_text_encode("", **run_encode_kwargs)
            for i in range(1,uncond.shape[1]-1):
                uncond[:,i] = uncond[:,0] if i%2==0 else uncond[:,-1]
            return uncond
        elif prompt == "<safety_concepts>": # could be used as negative prompt?
            uncond, _ = perform_text_encode("", **run_encode_kwargs)
            embeddings = globals().get("_safety_embeddings", None)
            if embeddings is None or not "concepts" in embeddings:
                return uncond
            else:
                # place between startoftext and endoftext vectors of uncond, rebuild batch dim
                return torch.stack([torch.stack([uncond[0][0]] + [x for x in embeddings["concepts"].clone().to(uncond.device, uncond.dtype)] + [uncond[0][1]])])
        elif prompt == "<safety_special>": # could be used as negative prompt?
            uncond, _ = perform_text_encode("", **run_encode_kwargs)
            embeddings = globals().get("_safety_embeddings", None)
            if embeddings is None or not "special" in embeddings:
                return uncond
            else:
                return torch.stack([torch.stack([uncond[0][0]] + [x for x in embeddings["special"].clone().to(uncond.device, uncond.dtype)] + [uncond[0][1]])])
        return None

    def apply_pad_tensor_to_embedding(in_tensor:torch.Tensor, pad_source:torch.Tensor, encapsulated=True):
        if encapsulated:
            # unpack and re-pack batch dimension of prompt
            return torch.stack([apply_pad_tensor_to_embedding(in_tensor[0],pad_source[0],False)])

        if in_tensor.shape[0] == pad_source.shape[0]:
            return in_tensor
        elif in_tensor.shape[0] > pad_source.shape[0]:
            tqdm.write(f"WARNING: Embed chunk too large when applying padding! Padding {in_tensor.shape} with {pad_source.shape} requested!")
            return in_tensor[:][:len(pad_source)]
        else:
            # fill as many vectors with data from the input tensor as possible. If the index is out of range for the input tensor, fill with the pad tensor. Rebuild batch dim.
            return torch.stack([in_tensor[i] if i in range(len(in_tensor)) else pad_vector for i, pad_vector in enumerate(pad_source)])

    def equalize_embedding_lengths(a:torch.Tensor, b:torch.Tensor):
        # if the embeddings are not of equal length, zero-pad the shorter of the two embeddings to make the shapes equal.
        if a is None and b is None:
            raise ValueError("Unable to equalize embedding lengths as both embeddings are None")
        a = a if a is not None else torch.zeros_like(b)
        b = b if b is not None else torch.zeros_like(a)
        a = a if not (b.shape[1] > a.shape[1]) else apply_pad_tensor_to_embedding(a,torch.zeros_like(b))
        b = b if not (a.shape[1] > b.shape[1]) else apply_pad_tensor_to_embedding(b,torch.zeros_like(a))
        return a,b

    def add_encoding_sum(current_sum:torch.Tensor, next_encoding:torch.Tensor, current_multiplier:float, next_multiplier:float, concat_direct_to_prev:bool):
        next_encoding *= next_multiplier
        # if the length of the current sum does not match the length of the next encoding, pad the shorter tensor.
        current_sum, next_encoding = equalize_embedding_lengths(current_sum, next_encoding)
        return (next_encoding if current_sum is None else current_sum + next_encoding), current_multiplier+next_multiplier

    def add_encoding_cat(current_sum:torch.Tensor, next_encoding:torch.Tensor, current_multiplier:float, next_multiplier:float, concat_direct_to_prev:bool):
        next_encoding *= next_multiplier
        # Remove startoftext and endoftext vectors if this is a direct concat.
        if concat_direct_to_prev and current_sum is not None:
            next_encoding = next_encoding[:,1:] # remove startoftext from new embedding
            current_sum = current_sum[:,:-1] # remove trailing endoftext from prev embedding
        #next_encoding = next_encoding if not concat_direct_to_next else next_encoding[:,:-1]
        # concat in dim 1 - skip batch dim. Return multiplier as 1 - no division by a scale sum should be performed in concat mode.
        return (next_encoding if current_sum is None else torch.cat([current_sum,next_encoding],dim=1)), 1

    @torch.no_grad()
    def encode_prompt(prompt, encoder_level_negative_prompts=False, clip_skip_layers=0, static_length=None, mix_mode_concat=False, extended_length_enforce_minimum=False, drop_zero_weight_encodings=CONCAT_DROP_ZERO_WEIGHT_EMBEDDINGS):
        # invalid (too small) lengths -> use model default
        static_length = tokenizer.model_max_length if (static_length is not None) and (static_length <= 2) else static_length
        perform_encode_kwargs = {
            "clip_skip_layers":clip_skip_layers,
            "pad_to_length":static_length,
            #"prefer_default_length":not dynamic_length_mode,
        }
        def io_data_template():
            return {"text_readback":[], "remaining_token_count":[], "weights":[], "clip_skip_layers":[], "tokens":[]}
        io_data = io_data_template()
        embeddings_list_uncond = []
        embeddings_list_prompt = []
        for raw_item in prompt:
            # create sequence of substrings followed by weight multipliers: substr,w,substr,w,... (w may be empty string - use default: 1, last trailing substr may be empty string)
            prompt_segments = re.split(";(-[\.0-9]+|[\.0-9]*\+?);", raw_item)
            if len(prompt_segments) == 1:
                # if the prompt does not specify multiple substrings with weights for mixing, run one encoding on default mode.
                new_embedding_positive, new_io_data = perform_text_encode(raw_item, **perform_encode_kwargs)
                new_embedding_negative, _ = perform_text_encode(DEFAULT_NEGATIVE_PROMPT, **perform_encode_kwargs)
            else:
                # print(f"Mixing {len(prompt_segments)//2} prompts with their respective weights")
                # perpare invalid segments list to desired shape after splitting
                if len(prompt_segments) % 2 == 1:
                    if prompt_segments[-1] == "":
                        # if last prompt in sequence is terminated by a weight, remove trailing empty prompt created by splitting.
                        prompt_segments = prompt_segments[:-1]
                    else:
                        # if last prompt in sequence is not empty, but lacks a trailing weight, append an empty weight.
                        prompt_segments.append("")

                # encoding tensors will be summed up with their respective multipliers (could be positive or negative).
                # Final tensor will be divided by (absolute!) sum of multipliers.
                encodings_sum_positive = None
                encodings_sum_negative = None
                multiplier_sum_positive = 0
                multiplier_sum_negative = 0
                new_io_data = io_data_template()
                concat_direct_to_prev = False
                for i, segment in enumerate(prompt_segments):
                    if i % 2 == 0:
                        next_encoding, next_io_data = perform_text_encode(segment, **perform_encode_kwargs)
                        for key in next_io_data:
                            # glue together the io_data sublist of all the prompts being mixed
                            new_io_data[key] += next_io_data[key]
                    else:
                        # get multiplier. 1 if no multiplier is present.
                        concat_direct_to_next = segment.endswith('+')
                        segment = segment[:-1] if concat_direct_to_next else segment # remove trailing '+' if present.
                        multiplier = 1 if segment == "" else float(segment)
                        new_io_data["text_readback"][-1] += f";{multiplier}{'+' if concat_direct_to_next else ''};"
                        # add either to positive, or to negative encodings & multipliers
                        is_negative_multiplier = multiplier < 0
                        # in dynamic length mode, concatenate prompts together instead of mixing. Otherwise, add new encodings to respective sum based on their multiplier.
                        if mix_mode_concat:
                            add_encoding = add_encoding_cat
                        else:
                            if not concat_direct_to_prev:
                                add_encoding = add_encoding_sum
                            else:
                                # if ;+; was passed in sum mode, perform a "normal concatenation" for this embedding join instead of a sum.
                                concat_direct_to_prev=False # disable direct concat for this concatenation, keep concat_direct_to_next to inform potential upcoming concats (next segment)
                                add_encoding = add_encoding_cat
                        if drop_zero_weight_encodings and multiplier == 0:
                            pass
                        elif is_negative_multiplier:
                            multiplier = abs(multiplier)
                            encodings_sum_negative, multiplier_sum_negative = add_encoding(encodings_sum_negative, next_encoding, multiplier_sum_negative, multiplier, concat_direct_to_prev)
                        else:
                            encodings_sum_positive, multiplier_sum_positive = add_encoding(encodings_sum_positive, next_encoding, multiplier_sum_positive, multiplier, concat_direct_to_prev)

                        # in case of direct concatenation, write flag for the next cycle.
                        concat_direct_to_prev = concat_direct_to_next

                if encoder_level_negative_prompts:
                    # old mixing mode. leaves uncond embeddings untouched, mixes negative prompts directly into the prompt embedding itself.
                    # if the length of the current sum does not match the length of the next encoding, pad the shorter tensor with zero-vectors to make the shapes equal.
                    encodings_sum_positive, encodings_sum_negative = equalize_embedding_lengths(encodings_sum_positive, encodings_sum_negative)
                    if encodings_sum_positive is None:
                        tqdm.write("WARNING: only negative multipliers for prompt mixing are present. Using '' as a placeholder for positive prompts!")
                        encodings_sum_positive, _ = perform_text_encode("", **perform_encode_kwargs)
                        multiplier_sum_positive = 1.0
                    new_text_embeddings = encodings_sum_positive / multiplier_sum_positive
                    if encodings_sum_negative is not None:
                        # compute difference (direction of change) of negative embeddings from positive embeddings. Move in opposite direction of negative embeddings from positive embeddings based on relative multiplier strength.
                        # this does not subtract negative prompts, but rather amplifies the difference from the positive embeddings to the negative embeddings
                        negative_embeddings_offset = (encodings_sum_negative / multiplier_sum_negative)
                        new_text_embeddings += ((new_text_embeddings) - (negative_embeddings_offset)) * (multiplier_sum_negative / multiplier_sum_positive)
                    # use mixed prompt tensor as prompt embedding
                    new_embedding_positive = new_text_embeddings
                    # use default, empty uncond embedding
                    new_embedding_negative, _ = perform_text_encode(DEFAULT_NEGATIVE_PROMPT, **perform_encode_kwargs)
                else:
                    if encodings_sum_positive is None:
                        # add an empty prompt for positive embeddings. Saving iodata of an empty placeholder is probably not necessary.
                        replacement_encoding_positive, additional_io_data = perform_text_encode("", **perform_encode_kwargs)
                        new_embedding_positive = replacement_encoding_positive
                    else:
                        new_embedding_positive = encodings_sum_positive / multiplier_sum_positive
                    if encodings_sum_negative is None:
                        # create default, empty uncond embedding. Skip iodata. It would not normally be saved for the empty uncond embedding.
                        replacement_encoding_negative, additional_io_data = perform_text_encode(DEFAULT_NEGATIVE_PROMPT, **perform_encode_kwargs)
                        new_embedding_negative = replacement_encoding_negative
                    else:
                        new_embedding_negative = (encodings_sum_negative / multiplier_sum_negative)

            embeddings_list_prompt.append(new_embedding_positive)
            embeddings_list_uncond.append(new_embedding_negative)
            for key in new_io_data:
                io_data[key].append(new_io_data[key])

        # if only one (sub)prompt was actually processed, use the io_data un-encapsulated
        if len(io_data["text_readback"]) == 1:
            for key in io_data:
                io_data[key] = io_data[key][0]

        embedding_lengths = [embedding.shape[1] for embedding in embeddings_list_prompt+embeddings_list_uncond]
        # if embedding sizes do not match, or if embeddings are shorter than an enforced minimum length, pad them wherever necessary.
        if (not all([el == embedding_lengths[0] for el in embedding_lengths])) or (extended_length_enforce_minimum and embedding_lengths[0]<tokenizer.model_max_length):
            max_embed_length = max(embedding_lengths)
            # if enforced, extend prompts to at least the default length
            max_embed_length = max_embed_length if not extended_length_enforce_minimum else max([tokenizer.model_max_length,max_embed_length])
            padded_uncond_embedding, _ = perform_text_encode("", **{k:v for k,v in perform_encode_kwargs.items() if not k in ["pad_to_length"]}, pad_to_length=max_embed_length)
            embeddings_list_prompt = [apply_pad_tensor_to_embedding(emb,padded_uncond_embedding) for emb in embeddings_list_prompt]
            embeddings_list_uncond = [apply_pad_tensor_to_embedding(emb,padded_uncond_embedding) for emb in embeddings_list_uncond]
        # stack list of text encodings to be processed back into a tensor
        text_embeddings = torch.cat(embeddings_list_prompt)
        # get the resulting batch size
        batch_size = len(text_embeddings)
        # create tensor of uncond embeddings for the batch size by stacking n instances of the singular uncond embedding
        uncond_embeddings = torch.cat(embeddings_list_uncond)
        # n*77*768 + n*77*768 -> 2n*77*768
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        if half_precision_latents:
                text_embeddings = text_embeddings.to(dtype=torch.float16)
        text_embeddings = text_embeddings.to(UNET_DEVICE)

        return text_embeddings, batch_size, io_data

    def get_gs_mult(gs_schedule:str, progress_factor:float):
        gs_mult = 1
        if gs_schedule is None or gs_schedule == "":
            pass
        elif gs_schedule == "sin": # half-sine (between 0 and pi; 0 -> 1 -> 0)
            gs_mult = np.sin(np.pi * progress_factor)
        elif gs_schedule == "isin": # inverted half-sine (1 -> 0 -> 1)
            gs_mult = 1.0 - np.sin(np.pi * progress_factor)
        elif gs_schedule == "fsin": # full sine (0 -> 1 -> 0 -> -1 -> 0) for experimentation
            gs_mult = np.sin(2*np.pi * progress_factor)
        elif gs_schedule == "anneal5": # rectified 2.5x full sine (5 bumps)
            gs_mult = np.abs(np.sin(2*np.pi * progress_factor*2.5))
        elif gs_schedule == "ianneal5": # 1- (rectified 2.5x full sine (5 bumps))
            gs_mult = 1 - np.abs(np.sin(2*np.pi * progress_factor*2.5))
        elif gs_schedule == "cos": # quarter-cos (between 0 and pi/2; 1 -> 0)
            gs_mult = np.cos(np.pi/2 * progress_factor)
        elif gs_schedule == "icos": # inverted quarter-cos (0 -> 1)
            gs_mult = 1.0 - np.cos(np.pi/2 * progress_factor)
        elif gs_schedule == "rand": # random multiplier in [0,1)
            gs_mult = np.random.rand()
        elif gs_schedule == "frand": # random multiplier, with negative values
            gs_mult = np.random.rand()*2-1
        else:
            # Could add a warning here.
            pass
        return gs_mult

    def get_scheduler(sched_name, high_beta=False):
        if high_beta:
            # scheduler default
            beta_start, beta_end = 0.0001, 0.02
        else:
            # stablediffusion default
            beta_start, beta_end = 0.00085, 0.012
        scheduler_params = {"beta_start":beta_start, "beta_end":beta_end, "beta_schedule":"scaled_linear", "num_train_timesteps":1000}
        if V_PREDICTION_MODEL:
            if not sched_name in V_PREDICTION_SCHEDULERS:
                tqdm.write(f"WARNING: A v_prediction model is running, but the selected scheduler {sched_name} is not listed as v_prediction enabled. THIS WILL YIELD BAD RESULTS! Switch to one of {V_PREDICTION_SCHEDULERS}")
            else:
                scheduler_params["prediction_type"] = "v_prediction"
        if "lms" == sched_name:
            scheduler = LMSDiscreteScheduler(**scheduler_params)
        elif "pndm" == sched_name:
            # "for some models like stable diffusion the prk steps can/should be skipped to produce better results."
            scheduler = PNDMScheduler(**scheduler_params, skip_prk_steps=True) # <-- pipeline default
        elif "ddim" == sched_name:
            scheduler = DDIMScheduler(**scheduler_params)
        elif "ipndm" == sched_name:
            scheduler = IPNDMScheduler(**scheduler_params)
        elif "euler" == sched_name:
            scheduler = EulerDiscreteScheduler(**scheduler_params)
        elif "euler_ancestral" == sched_name:
            scheduler = EulerAncestralDiscreteScheduler(**scheduler_params)
        elif "mdpms" == sched_name:
            scheduler = DPMSolverMultistepScheduler(**scheduler_params)
        elif "sdpms" == sched_name:
            scheduler = DPMSolverSinglestepScheduler(**scheduler_params)
        elif "kdpm2" == sched_name:
            scheduler = KDPM2DiscreteScheduler(**scheduler_params)
        elif "kdpm2_ancestral" == sched_name:
            scheduler = KDPM2AncestralDiscreteScheduler(**scheduler_params)
        elif "heun" == sched_name:
            scheduler = HeunDiscreteScheduler(**scheduler_params)
        elif "deis" == sched_name:
            scheduler = DEISMultistepScheduler(**scheduler_params)
        elif "unipc" == sched_name:
            scheduler = UniPCMultistepScheduler(**scheduler_params)

        else:
            raise ValueError(f"Requested unknown scheduler: {sched_name}")
        return scheduler

    # see diffusers PR: https://github.com/huggingface/diffusers/commit/12a232efa99d7a8c33f54ae515c5a3d6fc5c8f34
    def rescale_prediction(noise_pred:torch.Tensor, noise_pred_text:torch.Tensor, rescale_factor:float=0.66) -> torch.Tensor:
        def get_std(x:torch.Tensor) -> torch.Tensor:
            return x.std(dim=list(range(1, x.ndim)), keepdim=True)
        prediction_scale_factor = (get_std(noise_pred_text) / get_std(noise_pred))
        return noise_pred * (1-rescale_factor) + noise_pred*prediction_scale_factor*rescale_factor


    @torch.no_grad()
    def generate_segmented_exec(
            prompt=["art"], width=512, height=512, steps=50, gs=7.5, seed=None, sched_name="pndm", eta=0.0, eta_seed=None, high_beta=False,
            animate=False, init_image=None, img_strength=0.5, save_latents=False, gs_schedule=None, animate_pred_diff=True,
            encoder_level_negative_prompts=False, clip_skip_layers=0, static_length=None, mix_mode_concat=False,
            controlnet:ControlNetModel=None,controlnet_input=None,controlnet_strength=1.0,controlnet_preprocessor=None,controlnet_strength_scheduler=None,
            guidance_rescale=0.66, *args, **kwargs
        ):
        gc.collect()
        if "cuda" in [IO_DEVICE, DIFFUSION_DEVICE] or ("meta" in [IO_DEVICE, DIFFUSION_DEVICE] and OFFLOAD_EXEC_DEVICE == "cuda"):
            torch.cuda.empty_cache()
        START_TIME = time()

        SUPPLEMENTARY = {
            "io" : {
                "width" : 0,
                "height" : 0,
                "steps" : steps,
                "gs" : gs,
                "gs_sched" : gs_schedule,
                "text_readback" : "",
                "nsfw" : False,
                "remaining_token_count" : 0,
                "time" : 0.0,
                "image_sequence" : [],
                "seed" : 0,
                "eta_seed" : 0,
                "eta" : eta,
                "sched_name" : None,
                "weights" : [],
                "unet_model" : f"{models_local_dir} (local)" if using_local_unet else model_id,
                "vae_model" : f"{models_local_dir} (local)" if using_local_vae else model_id,
                "encoder_level_negative" : encoder_level_negative_prompts,
                "clip_skip_layers" : [clip_skip_layers], # will be overridden by encode_prompt
                "static_length" : static_length,
                "concat" : mix_mode_concat,
                "LoRA" : None if ACTIVE_LORA is None else f"{ACTIVE_LORA}: {ACTIVE_LORA_WEIGHT}",
                "controlnet": None if None in [controlnet, controlnet_input] else f"{getattr(controlnet, '_MODEL_NAME', 'unlabeled')}: {controlnet_strength}",
                "controlnet_scheduler": controlnet_strength_scheduler,
                "processed_controlnet_input": None,
                "guidance_rescale": guidance_rescale if guidance_rescale > 0 else None,
            },
            "latent" : {
                "final_latent" : None,
                "latent_sequence" : [],
            },
        }
        if isinstance(ACTIVE_LORA,list) and isinstance(ACTIVE_LORA_WEIGHT,list) and len(ACTIVE_LORA)==len(ACTIVE_LORA_WEIGHT):
            try: SUPPLEMENTARY["io"]["LoRA"] = [f"{lora.rsplit('.',1)[0]}: {weight}" for lora,weight in zip(ACTIVE_LORA,ACTIVE_LORA_WEIGHT)]
            except Exception as e: tqdm.write(f"Failed to create IOData LoRA info from {ACTIVE_LORA} with weights {ACTIVE_LORA_WEIGHT}")

        # truncate incorrect input dimensions to a multiple of 64
        if isinstance(prompt, str):
            prompt = [prompt]
        height = int(height/8.0)*8
        width = int(width/8.0)*8
        SUPPLEMENTARY["io"]["height"] = height
        SUPPLEMENTARY["io"]["width"] = width
        # torch.manual_seed: Value must be within the inclusive range [-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff], negative values remapped to positive
        # 0xffff_ffff_ffff_ffff == 18446744073709551615 theoretical max. In the EXTREMELY unlikely event that something very close to the max value is picked in sequential mode, leave some headroom. (>50k in this case.)
        seed = seed if seed is not None else random.randint(1, 18446744073709500000)
        eta_seed = eta_seed if eta_seed is not None else random.randint(1, 18446744073709500000)
        SUPPLEMENTARY["io"]["seed"] = seed
        SUPPLEMENTARY["io"]["eta_seed"] = eta_seed
        generator_eta = torch.manual_seed(eta_seed)
        if not IO_DEVICE == "meta":
            generator_unet = torch.Generator(IO_DEVICE).manual_seed(seed)
        else:
            generator_unet = torch.Generator("cpu").manual_seed(seed)

        sched_name = sched_name.lower().strip()
        SUPPLEMENTARY["io"]["sched_name"]=sched_name

        text_embeddings, batch_size, io_data = encode_prompt(prompt, encoder_level_negative_prompts, clip_skip_layers, static_length, mix_mode_concat)
        for key in io_data:
            SUPPLEMENTARY["io"][key] = io_data[key]

        scheduler = get_scheduler(sched_name, high_beta)
        # set timesteps. Also offset, as pipeline does this
        scheduler_offset = 1 if "offset" in set(inspect.signature(scheduler.set_timesteps).parameters.keys()) else 0
        scheduler.set_timesteps(steps, **({"offset":scheduler_offset} if scheduler_offset>0 else {}))

        starting_timestep = 0
        # initial noise latents
        if init_image is None or (isinstance(init_image, list) and len(init_image) == 0):
            latents = torch.randn((batch_size, unet.config.in_channels, height // 8, width // 8), device=IO_DEVICE, generator=generator_unet)
            if half_precision_latents:
                latents = latents.to(dtype=torch.float16)
            latents = latents.to(UNET_DEVICE)
        else:
            SUPPLEMENTARY["io"]["strength"] = img_strength
            if not isinstance(init_image, list):
                init_images = [init_image]
            else:
                init_images = init_image
            init_latents_list = []
            for init_image in init_images:
                if init_image.size != (width,height):
                    init_image = init_image.resize((width,height), resample=Image.Resampling.LANCZOS)
                image_processing = np.array(init_image).astype(np.float32) / 255.0
                image_processing = image_processing[None].transpose(0, 3, 1, 2)
                init_image_tensor = 2.0 * torch.from_numpy(image_processing) - 1.0
                init_latents = vae_with_failover(vae, init_image_tensor.to(IO_DEVICE), decode=False).latent_dist.sample()
                # apply inverse scaling
                init_latents = 0.18215 * init_latents
                init_latents_list.append(init_latents)
            # if the init latent batch is smaller than the actual batch size, expand it.
            if len(init_latents_list) < batch_size:
                init_latents_list *= (batch_size//len(init_latents_list)) + (1 if (batch_size%len(init_latents_list)) > 0 else 0)
            # reduce init latent batch to actual batch size if oversized.
            init_latents = torch.cat(init_latents_list[:batch_size])
            init_timestep = int(steps * img_strength) + scheduler_offset
            init_timestep = min(init_timestep, steps)
            timesteps = scheduler.timesteps[-init_timestep]
            timesteps = torch.stack([timesteps] * batch_size).to(dtype=torch.long, device=IO_DEVICE)
            noise = torch.randn(init_latents.shape, generator=generator_unet, device=IO_DEVICE)
            init_latents = scheduler.add_noise(init_latents, noise, timesteps)
            if half_precision_latents:
                init_latents = init_latents.to(dtype=torch.float16)
            latents = init_latents.to(UNET_DEVICE)
            starting_timestep = max(steps - init_timestep + scheduler_offset, 0)

        if hasattr(scheduler, "init_noise_sigma"):
            latents *= scheduler.init_noise_sigma

        if controlnet_input is not None:
            controlnet_input, preprocessed_image = controlnet_preprocess(controlnet_input, width, height, controlnet_preprocessor)
            SUPPLEMENTARY["io"]["processed_controlnet_input"] = preprocessed_image
            controlnet_input = torch.cat([controlnet_input]*len(latents)*2)
            if controlnet is None:
                tqdm.write("Warning! Controlnet input provided, but no controlnet found!")
                controlnet_input = None
        elif controlnet is not None:
            pass # as long as controlnet_input is None, the controlnet won't actually be computed or applied
            # tqdm.write("Warning! Controlnet provided, but no controlnet input found!")

        # denoising loop!
        progress_bar = tqdm([x for x in enumerate(scheduler.timesteps[starting_timestep:])], position=1)
        unet_sequential_batch = False
        for i, t in progress_bar:
            if animate and not animate_pred_diff:
                SUPPLEMENTARY["latent"]["latent_sequence"].append(latents.cpu().clone())
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            if hasattr(scheduler, "scale_model_input"):
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            progress_factor = i/len(scheduler.timesteps[starting_timestep:])
            # predict the noise residual
            with torch.no_grad():
                cast_device = UNET_DEVICE #if UNET_DEVICE != "meta" else OFFLOAD_EXEC_DEVICE
                cast_unet_args = {"device":cast_device, "dtype":unet.dtype}
                down_block_residuals, mid_block_residual, incompatible = apply_controlnet(controlnet, controlnet_input, controlnet_strength, controlnet_strength_scheduler, text_embeddings, t, latent_model_input, progress_factor, cast_unet_args)
                if incompatible:
                    tqdm.write(f"Dimension mismatch between controlnet and diffusion UNET! Versions may be mismatched between SD1.x and SD2.x! Disabling controlnet for this diffusion.")
                    controlnet_input = None
                    SUPPLEMENTARY["io"]["controlnet"] = None
                if not unet_sequential_batch:
                    try:
                        noise_pred = unet(
                            latent_model_input.to(**cast_unet_args), t.to(**cast_unet_args), encoder_hidden_states=text_embeddings.to(**cast_unet_args),
                            down_block_additional_residuals=down_block_residuals, mid_block_additional_residual=mid_block_residual,
                        )["sample"]
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            unet_sequential_batch = True
                        else:
                            raise e
                if unet_sequential_batch:
                    # re-check the condition. Will now be True if batching failed.
                    def batch_idx(x, i):
                        return None if x is None else torch.stack([x[i]])
                    noise_pred = torch.cat([unet(
                        batch_idx(latent_model_input,i).to(**cast_unet_args), t.to(**cast_unet_args), encoder_hidden_states=batch_idx(text_embeddings,i).to(**cast_unet_args),
                        down_block_additional_residuals=None if down_block_residuals is None else batch_idx(down_block_residuals,i).to(**cast_unet_args),
                        mid_block_additional_residual=None if mid_block_residual is None else mid_block_residual.to(**cast_unet_args),
                    )["sample"] for i in range(len(latent_model_input))])

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            gs_mult = get_gs_mult(gs_schedule, progress_factor)
            progress_bar.set_description(f"gs={gs*gs_mult:.3f}{';seq_batch' if unet_sequential_batch else ''}")
            noise_pred = noise_pred_uncond + gs * (noise_pred_text - noise_pred_uncond) * gs_mult
            if isinstance(guidance_rescale, float) and guidance_rescale > 0:
                noise_pred = rescale_prediction(noise_pred, noise_pred_text, guidance_rescale)

            if animate and animate_pred_diff:
                animate_latents = ((latents-noise_pred)*(1-(i/len(scheduler.timesteps))**2) + latents*(i/len(scheduler.timesteps))**2).cpu().clone()
                SUPPLEMENTARY["latent"]["latent_sequence"].append(animate_latents)
            del noise_pred_uncond, noise_pred_text
            # compute the previous noisy sample x_t -> x_t-1
            scheduler_additional_args = {}
            if isinstance(scheduler, DDIMScheduler):
                scheduler_additional_args = {"eta":eta, "generator":generator_eta}
            latents = scheduler.step(noise_pred, t, latents, **scheduler_additional_args)["prev_sample"]

            del noise_pred

        # free up some now unused memory before attempting VAE decode!
        del latent_model_input, scheduler, text_embeddings, t
        gc.collect()
        if "cuda" in [IO_DEVICE, DIFFUSION_DEVICE]:
            torch.cuda.empty_cache()

        # save last latent before scaling!
        if save_latents:
            SUPPLEMENTARY["latent"]["final_latent"] = latents.clone().cpu()
        if animate:
            SUPPLEMENTARY["latent"]["latent_sequence"].append(latents.clone().cpu())
        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        latents = latents.to(IO_DEVICE)
        latents = latents.to(vae.dtype)
        image = vae_with_failover(vae, latents, decode=True)

        # latents are decoded and copied (if requested) by now.
        del latents

        def to_pil(image, should_rate_nsfw:bool=True):
            image = (image.sample / 2 + 0.5).clamp(0, 1)
            #image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            images = (image * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]
            if should_rate_nsfw:
                has_nsfw = rate_nsfw(pil_images)
            else:
                has_nsfw = False
            return pil_images, has_nsfw

        pil_images, has_nsfw = to_pil(image)
        SUPPLEMENTARY["io"]["nsfw"] = has_nsfw
        if animate:
            # swap places between UNET and VAE to speed up large batch decode. then swap back.
            try:
                unet.to(IO_DEVICE)
                vae.to(DIFFUSION_DEVICE)
                torch.cuda.synchronize()
            except NotImplementedError:
                pass # when offloaded to cpu, model switching will fail (copy out of meta tensor)
            with torch.no_grad():
                # process items one-by-one to avoid overfilling VRAM with a batch containing all items at once.
                SUPPLEMENTARY["io"]["image_sequence"] = [to_pil(vae_with_failover(vae,(item*(1 / 0.18215)).to(DIFFUSION_DEVICE,vae.dtype)), False)[0] for item in tqdm(SUPPLEMENTARY["latent"]["latent_sequence"], position=0, desc="Decoding animation latents")]
            try:
                torch.cuda.synchronize()
                vae.to(IO_DEVICE)
                unet.to(DIFFUSION_DEVICE)
            except NotImplementedError:
                pass
        SUPPLEMENTARY["io"]["time"] = time() - START_TIME

        return pil_images, SUPPLEMENTARY

    def apply_controlnet(controlnet, controlnet_input, controlnet_strength, controlnet_strength_scheduler, text_embeddings, t, latent_model_input, progress_factor, cast_unet_args):
        incompatible = False
        if controlnet_input is not None:
            controlnet_mult = get_gs_mult(controlnet_strength_scheduler, progress_factor)
            cast_controlnet_args = {"device":controlnet.device, "dtype":torch.float16}
            try:
                down_block_residuals, mid_block_residual = controlnet(latent_model_input.to(**cast_controlnet_args),t.to(**cast_controlnet_args),encoder_hidden_states=text_embeddings.to(**cast_controlnet_args),controlnet_cond=controlnet_input.to(**cast_controlnet_args),return_dict=False)
                down_block_residuals = [down_block_res_sample * controlnet_strength*controlnet_mult for down_block_res_sample in down_block_residuals]
                mid_block_residual *= controlnet_strength*controlnet_mult
                down_block_residuals = [d.to(**cast_unet_args) for d in down_block_residuals]
                mid_block_residual = mid_block_residual.to(**cast_unet_args)
            except RuntimeError as e:
                if f"The size of tensor a" in str(e) and "must match the size of tensor b" in str(e):
                    tqdm.write(str(e))
                    down_block_residuals=None
                    mid_block_residual=None
                    incompatible = True
                else:
                    raise e
        else:
            down_block_residuals=None
            mid_block_residual=None
        return down_block_residuals,mid_block_residual, incompatible

    def cleanup_memory():
        gc.collect()
        if "cuda" in [IO_DEVICE, DIFFUSION_DEVICE]:
            torch.cuda.empty_cache()

    # can optionally move the unet out of the way to make space
    @torch.no_grad()
    def vae_with_failover(vae, latents:torch.Tensor, decode=True):
        results = vae_with_failover_worker(vae, latents, decode=decode)
        if results is None: # fully exit the previous context instead of recursively calling to ensure that memory is properly released.
            return vae_with_failover_worker(vae, latents, True, decode=decode)
        return results
    @torch.no_grad()
    def vae_with_failover_worker(vae, latents:torch.Tensor, minimize_memory=False, decode=True):
        cleanup_memory()
        try:
            if minimize_memory:
                was_slicing, was_tiling = vae.use_slicing, vae.use_tiling
                vae.enable_slicing() # should take care of potential OOMs caused by large batches
                vae.enable_tiling() # should take care of potential OOMs caused by large resolution
            results = vae.decode(latents.to(vae.device)) if decode else vae.encode(latents.to(vae.device))
            if minimize_memory:
                if not was_slicing:
                    vae.disable_slicing()
                if not was_tiling:
                    vae.disable_tiling()
            return results
        except RuntimeError as e:
            if 'out of memory' in str(e):
                if not minimize_memory:
                    # recursively calling with slicing/tiling enabled after an OOM has occurred does not yield the expected memory reduction. Potential issue with releasing CUDA memory?
                    # exit local context, prompting the wrapper to call again with tiling and slicing enabled.
                    return None
                if vae.device != "cpu":
                    tqdm.write(f"VAE {'decode' if decode else 'encode'} moved to CPU.")
                    return vae_code_cpu(vae, latents, decode=decode)
                else:
                    raise RuntimeError(f"VAE {'decode' if decode else 'encode'} ran out of memory, with VAE already on CPU: {e}")
            else:
                # there was some RuntimeError, but it wasn't OOM.
                raise e

    @torch.no_grad()
    def vae_code_cpu(vae, latents:torch.Tensor, decode=True):
        cleanup_memory()
        prev_device = vae.device
        try:
            vae = vae.to(device="cpu")
            latents = latents.to(device="cpu")
            image_out = vae.decode(latents) if decode else vae.encode(latents)
            vae = vae.to(device=prev_device)
            return image_out
        except NotImplementedError as e:
            if "No operator found" in str(e):
                vae.set_use_memory_efficient_attention_xformers(False,None)
                result = vae_code_cpu(vae, latents, decode=decode)
                vae.set_use_memory_efficient_attention_xformers(True and XFORMERS_ENABLED,None)
                vae = vae.to(device=prev_device) # the recursive call will find the VAE already on CPU (error will have occurred during vae.decode), therefore it must be returned to the original device here.
                return result
            if not ("Cannot copy out of meta tensor" in str(e)):
                tqdm.write(f"VAE {'decode' if decode else 'encode'} failed due to unknown implementation issue: {e}. Attempting decode with stand-in VAE")
            # vae is offloaded through accelerate, or XFormers attention is enabled. use a temporary stand-in vae.
            new_vae = load_models(vae_only=True)
            new_vae.set_use_memory_efficient_attention_xformers(False,None)
            new_vae = new_vae.to(device="cpu")
            latents = latents.to(device="cpu")
            image_out = new_vae.decode(latents) if decode else new_vae.encode(latents)
            del new_vae
            cleanup_memory()
            return image_out

    @torch.no_grad()
    def vae_encode_cpu(vae, input):
        cleanup_memory()
        prev_device = vae.device
        try:
            vae = vae.to(device="cpu")
            out = vae.decode(input)
            vae = vae.to(device=prev_device)
            return out
        except NotImplementedError as e:
            if "No operator found" in str(e):
                vae.set_use_memory_efficient_attention_xformers(False,None)
                result = vae_encode_cpu(vae, input)
                vae.set_use_memory_efficient_attention_xformers(True and XFORMERS_ENABLED,None)
                vae = vae.to(device=prev_device) # the recursive call will find the VAE already on CPU (error will have occurred during vae.decode), therefore it must be returned to the original device here.
                return result
            if not ("Cannot copy out of meta tensor" in str(e)):
                tqdm.write(f"VAE decode failed due to unknown implementation issue: {e}. Attempting decode with stand-in VAE")
            # vae is offloaded through accelerate, or XFormers attention is enabled. use a temporary stand-in vae.
            new_vae = load_models(vae_only=True)
            new_vae.set_use_memory_efficient_attention_xformers(False,None)
            new_vae = new_vae.to(device="cpu")
            out = new_vae.encode(input)
            del new_vae
            cleanup_memory()
            return out

    # TODO: Consider upgrading this to use a SD-latent-upscaler-like resize?
    def generate_two_step_wrapper(second_pass_resize=1, second_pass_steps=0, second_pass_use_controlnet=False, **exec_args):
        if not all([isinstance(second_pass_steps, int), second_pass_steps>0, isinstance(second_pass_resize, float), second_pass_resize>1]):
            # if two step generation is not specified, simply pass through.
            return generate_segmented_exec(**exec_args)

        if second_pass_use_controlnet and (exec_args.get("controlnet",None) is None):
            tqdm.write("Warning: Second pass was set to use controlnet, but no controlnet was specified! Falling back to img2img.")
            second_pass_use_controlnet = False

        out, SUPPLEMENTARY = generate_segmented_exec(**exec_args)
        initial_time = SUPPLEMENTARY['io']['time']
        initial_sequence = SUPPLEMENTARY["io"].get("image_sequence", [])
        init_iodata = {f"init_{k}":v for k,v in SUPPLEMENTARY["io"].items() if k in ["width","height","steps"]}
        # use round here instead of truncated int-conversion: We want to be as close to the original aspect ratio as possible to minimize image distortion
        second_pass_size = {k:round(v*second_pass_resize/8)*8 for k,v in exec_args.items() if k in ["height", "width"]}
        out = [x.resize((second_pass_size["width"],second_pass_size["height"]), resample=Image.Resampling.LANCZOS) for x in out]
        sharp_out = []
        for img in out:
            enhancer_sharpen = ImageEnhance.Sharpness(img)
            img = enhancer_sharpen.enhance(second_pass_resize)
            sharp_out.append(img)
        out = sharp_out
        exec_args = {**exec_args, **second_pass_size, "steps":second_pass_steps}
        if not second_pass_use_controlnet:
            exec_args["init_image"] = out
        else:
            exec_args["controlnet_input"] = out if isinstance(out, Image.Image) else out[0]
        out, SUPPLEMENTARY = generate_segmented_exec(**exec_args)
        SUPPLEMENTARY['io']['time'] += initial_time
        if len(initial_sequence) > 0 and len(SUPPLEMENTARY["io"].get("image_sequence", [])) > 0:
            final_size = SUPPLEMENTARY["io"]["image_sequence"][0][0].size # take the size of the first image in the sequence of image batches
            SUPPLEMENTARY["io"]["image_sequence"] = [[x.resize(final_size, resample=Image.Resampling.LANCZOS) for x in batch_idx] for batch_idx in initial_sequence] + SUPPLEMENTARY["io"]["image_sequence"]
        elif len(initial_sequence) > 0:
            # it really should not be possible to reach this case
            SUPPLEMENTARY["io"]["image_sequence"] = initial_sequence
        SUPPLEMENTARY["io"] = {**SUPPLEMENTARY["io"], **init_iodata}
        return out, SUPPLEMENTARY

    @torch.no_grad()
    def generate_segmented_wrapper(prompt=["art"], save_latents=False, sequential=False, **kwargs):
        # any additional exec args to be forwarded not captured by **kwargs must be added explicitly
        additional_exec_args = {"save_latents":save_latents,}
        exec_args = {**kwargs, **additional_exec_args}
        if not sequential:
            try:
                return generate_two_step_wrapper(prompt=prompt,**exec_args)
            except RuntimeError as e:
                if 'out of memory' in str(e) and len(prompt) > 1:
                    # if an OOM error occurs with batch size >1 in parallel, retry sequentially.
                    tqdm.write("Generating batch in parallel ran out of memory! Switching to sequential generation! To run sequential generation from the start, specify '-seq'.")
                    sequential=True
                else:
                    # if the error was not OOM due to batch size, raise it.
                    raise e
        if sequential:
            if exec_args["animate"]:
                tqdm.write("--animate is currently not supported for sequential generation. Disabling.")
                exec_args["animate"] = False
            # if sequential is requested, run items one-by-one
            out = []
            seeds = []
            eta_seeds = []
            SUPPLEMENTARY = None
            try:
                for prompt_item in tqdm(prompt, position=0, desc="Sequential Batch"):
                    # have deterministic seeds in sequential mode: every index after the first will be assigned n+1 as a seed.
                    exec_args["seed"] = exec_args["seed"] if len(seeds) == 0 else seeds[-1] + 1
                    exec_args["eta_seed"] = exec_args["eta_seed"] if len(eta_seeds) == 0 else eta_seeds[-1] + 1
                    new_out, new_supplementary = generate_two_step_wrapper(prompt=prompt_item, **exec_args)
                    # reassemble list of outputs
                    out += new_out
                    seeds.append(new_supplementary['io']['seed'])
                    eta_seeds.append(new_supplementary['io']['eta_seed'])
                    if SUPPLEMENTARY is None:
                        SUPPLEMENTARY = new_supplementary
                        SUPPLEMENTARY['io']['text_readback'] = [SUPPLEMENTARY['io']['text_readback']]
                        SUPPLEMENTARY['io']['weights'] = [SUPPLEMENTARY['io']['weights']]
                    else:
                        if save_latents:
                            # reassemble stack tensor of final latents
                            SUPPLEMENTARY['latent']['final_latent'] = torch.stack([x for x in SUPPLEMENTARY['latent']['final_latent']] + [new_supplementary['latent']['final_latent'][0]])
                        # reassemble other supplementary return data
                        SUPPLEMENTARY['io']['text_readback'] += [new_supplementary['io']['text_readback']]
                        SUPPLEMENTARY['io']['remaining_token_count'] += new_supplementary['io']['remaining_token_count']
                        SUPPLEMENTARY['io']['time'] += new_supplementary['io']['time']
                        SUPPLEMENTARY['io']['weights'] += [new_supplementary['io']['weights']]
                        SUPPLEMENTARY['io']['nsfw'] = SUPPLEMENTARY['io']['nsfw'] or new_supplementary['io']['nsfw']
                    SUPPLEMENTARY['io']['seed'] = seeds
                    SUPPLEMENTARY['io']['eta_seed'] = eta_seeds
            except KeyboardInterrupt:
                pass
            SUPPLEMENTARY['io']['sequential'] = True
            return out, SUPPLEMENTARY
    return generate_segmented_wrapper

# see: https://huggingface.co/sd-concepts-library | adapted from the "Inference Colab" notebook
# load learned embeds (textual inversion) into an encoder and tokenizer
@torch.no_grad()
def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, target_device="cpu", token=None):
    global PROMPT_SUBST
    loaded_learned_embeds = torch.load(learned_embeds_path, map_location=target_device)
    if isinstance(loaded_learned_embeds, dict) and all([k in loaded_learned_embeds for k in ["string_to_param", "name"]]):
        # .pt-style concepts
        token = loaded_learned_embeds["name"]
        param = loaded_learned_embeds["string_to_param"]
        embeds = param[list(param.keys())[0]]
    else:
        # sd-concepts-library-style concepts
        trained_token = list(loaded_learned_embeds.keys())[0]
        token = token if token is not None else trained_token
        embeds = loaded_learned_embeds[trained_token]

    # convert [768] Tensor to [1,768]
    if len(embeds.shape) == 1:
        embeds = torch.stack([embeds])
    # cast to dtype of text_encoder
    dtype = text_encoder.get_input_embeddings().weight.dtype
    embeds.to(dtype)
    embeds.requires_grad = False

    # store original token for substitution dict
    original_token = token
    # remove leading and trailing <> if present.
    token = token[1 if token.startswith("<") else None:-1 if token.endswith(">") else None]
    tokens = [f"<{token}>"] if embeds.shape[0] == 1 else [f"<{token}{i}>" for i in range(embeds.shape[0])]
    token_subst_value = "".join(tokens)

    # add the token(s) in tokenizer
    for (token, embed) in zip(tokens, embeds):
        try:
            encoder_shape = text_encoder.get_input_embeddings().weight.data[0].shape
            if not encoder_shape == embed.shape:
                if encoder_shape[0] in [768, 1024] and embed.shape[0] in [768,1024]:
                    sd1_clip = "SD_1.x" #"CLIP-ViT-L/14, SD_1.x"
                    sd2_clip = "SD_2.x" #"OpenCLIP-ViT/H, SD_2.x"
                    raise RuntimeError(f"Text encoder: {sd1_clip if encoder_shape[0] == 768 else sd2_clip}, embed: {sd1_clip if embed.shape[0] == 768 else sd2_clip}")
                raise RuntimeError(f"Incompatible: embed shape {embed.shape} does not match text encoder shape {text_encoder.get_input_embeddings().weight.data[0].shape}")
            num_added_tokens = tokenizer.add_tokens(token)
            if num_added_tokens == 0:
                # simply attempt to add the token with a number suffix
                for i in range(0, 256):
                    if num_added_tokens == 0:
                        num_added_tokens = tokenizer.add_tokens(f"{token}{i}")
                    else:
                        break
                if num_added_tokens == 0:
                    print(f"WARNING: Unable to add token {token} to tokenizer. Too many instances? Skipping addition!")
                    continue
            # resize the token embeddings
            text_encoder.resize_token_embeddings(len(tokenizer))
            # get the id for the token and assign the embed
            token_id = tokenizer.convert_tokens_to_ids(token)
            text_encoder.get_input_embeddings().weight.data[token_id] = embed
        except RuntimeError as e:
            print(f" (incompatible: {token}) {e}")
            return
            #print_exc()

    if not original_token in PROMPT_SUBST:
        PROMPT_SUBST[original_token] = token_subst_value
        print(f" {original_token}{'' if len(tokens)==1 else ' (<-'+str(len(tokens))+' tokens)'}", end="")
        return
    else:
        for i in range(256):
            alt_token = f"{original_token}{i}"
            if not alt_token in PROMPT_SUBST:
                PROMPT_SUBST[f"{original_token}{i}"] = token_subst_value
                print(f" {original_token}{i}{'' if len(tokens)==0 else '('+str(len(tokens))+' tokens)'}", end="")
                return
        # if for loop failed to return
        print(f"Failed to find index for substitution item for token {original_token}! Too many instances?")

# can be called with perform_save=False to generate output image (grid_image when multiple inputs are given) and metadata
@torch.no_grad()
def save_output(p, imgs, argdict, perform_save=True, latents=None, display=False, data_override:dict=None):
    argdict = None if argdict is None else {k:v for k,v in argdict.items() if not k in ["processed_controlnet_input"]}
    if isinstance(imgs, list):
        if len(imgs) > 1:
            multiple_images = True
            img = image_autogrid(imgs)
        else:
            multiple_images = False
            img = imgs[0]
    else:
        multiple_images = False
        img = imgs

    TIMESTAMP = datetime.today().strftime('%Y%m%d%H%M%S_%f') # microsecond suffix to prevent overwrites when re-encoding individual images of same batch
    metadata = PngInfo()

    if data_override is None:
        grid_latent_string = tensor_to_encoded_string(latents)
        # keep original shape: tensor[image_index][...], by creating tensor with only one index
        if multiple_images:
            if latents is None:
                item_latent_strings = [""]*len(imgs)
            else:
                item_latent_strings = [tensor_to_encoded_string(torch.stack([x])) for x in latents]
        prompts_for_filename, pm = cleanup_str(p)
        prompts_for_metadata = pm.replace("\n","\\n")

        argdict_str = str(argdict).replace('\n', '\\n')

        item_metadata_list = []
        if multiple_images:
            for (new_img, i) in zip(imgs, range(len(imgs))):
                _, item_pm = cleanup_str(p[i])
                item_prompt_for_metadata = item_pm.replace("\n","\\n")
                item_metadata = PngInfo()
                item_metadata.add_text("prompt", str(p[i]))
                item_metadata.add_text("prompt_data", f"{item_prompt_for_metadata}\n{argdict_str}")
                item_metadata.add_text("index", str(i))
                item_metadata.add_text("latents", item_latent_strings[i])
                filepath_noext = os.path.join(INDIVIDUAL_OUTPUTS_DIR, f"{TIMESTAMP}_{prompts_for_filename}_{i}")
                if perform_save:
                    new_img.save(filepath_noext+".png", pnginfo=item_metadata)
                item_metadata_list.append(item_metadata)

        metadata.add_text("prompt", str(collapse_representation(p)))
        metadata.add_text("prompt_data", f"{prompts_for_metadata}\n{argdict_str}")
        metadata.add_text("latents", grid_latent_string)
    else:
        item_metadata_list=[] # placeholder for return
        prompts_for_filename, prompts_for_metadata = cleanup_str(data_override.pop("filename",""))
        argdict = argdict if argdict is not None else data_override
        argdict.pop("latents", None)
        for k,v in data_override.items():
            metadata.add_text(str(k), str(v))

    base_filename = f"{TIMESTAMP}_{prompts_for_filename}"
    filepath_noext = os.path.join(OUTPUTS_DIR, f'{base_filename}')

    if perform_save:
        img.save(filepath_noext+".png", pnginfo=metadata)
        tqdm.write(f"Saved {filepath_noext}.png {argdict}")

    if display:
        show_image(img)
    if perform_save:
        with codecs.open(os.path.join(UNPUB_DIR, f"{base_filename}.txt"), mode="w", encoding="cp1252", errors="ignore") as f:
            f.write(prompts_for_metadata.replace("\n","\\n"))
            f.write("\n")
            f.write(str(argdict).replace("\n","\n"))
    if not perform_save:
        if not "NOT_SAVED_WARNING_PRINTED" in globals():
            global NOT_SAVED_WARNING_PRINTED
            NOT_SAVED_WARNING_PRINTED = True
            tqdm.write("perform_save set to False, results not saved to local disk.")

    return img, metadata, item_metadata_list

# cleanup string for use in a filename, extract string from list/tuple
def cleanup_str(input):
    if isinstance(input, (list, Tuple)) and len(input) == 1:
        s = str(input[0])
    else:
        s = str(input)
    new_string = "".join([char if char.isalnum() else "_" for char in s])
    # limit to something reasonable
    while "__" in new_string:
        new_string = new_string.replace("__","_")
    return new_string[:64], s

def show_image(img, title=None):
    try:
        cv2.imshow(CV2_TITLE if title is None else title, cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
        cv2.waitKey(5)
    except:
        pass

# function to create one image containing all input images in a grid.
# currently not intended for images of differing sizes.
def image_autogrid(imgs, fixed_rows=None, fill_color=None, separation=None, frame_grid=False) -> Image.Image:
    try:
        assert fill_color is not None
        fill_color = ImageColor.getrgb(fill_color)
    except Exception:
        fill_color = GRID_IMAGE_BACKGROUND_COLOR
    separation = try_int(separation, GRID_IMAGE_SEPARATION)
    if fixed_rows is not None:
        rows = fixed_rows
        cols = math.ceil(len(imgs)/fixed_rows)
    elif len(imgs) == 3:
        cols=3
        rows=1
    else:
        side_len = math.sqrt(len(imgs))
        # round up cols from square root, attempt to round down rows
        # if required to actually fit all images, both cols and rows are rounded up.
        cols = math.ceil(side_len)
        rows = math.floor(side_len)
        if (rows*cols) < len(imgs):
            rows = math.ceil(side_len)
    # get grid item size from first image
    w, h = imgs[0].size
    # add separation to size between images as 'padding'
    w += separation * (1 if not frame_grid else 2)
    h += separation * (1 if not frame_grid else 2)
    # remove one image separation size from the overall size (no added padding after the final row/col)
    grid = Image.new('RGB', size=(cols*w-(separation if not frame_grid else 0), rows*h-(separation if not frame_grid else 0)), color=fill_color)
    start_offset = 0 if not frame_grid else separation
    for i, img in enumerate(imgs):
        grid.paste(img, box=(start_offset+i%cols*w, start_offset+i//cols*h))
    return grid

@torch.no_grad()
def get_safety_checker(device="cpu", safety_model_id = "CompVis/stable-diffusion-safety-checker", cpu_offloading=False):
    # backwards compatibility
    return SafetyChecker(device=device, safety_model_id=safety_model_id, cpu_offloading=cpu_offloading)

class SafetyChecker():
    # Reverse maps of (segmentation model) tokenized representations for a given safety checker "bad concept" embedding. Used to inform the segmentation model when applying masked blur processing.
    # Obtained via CLIP comparison of a large dictionary against given concept embeddings. Resulting concept labels line up with https://github.com/LAION-AI/CLIP-based-NSFW-Detector
    _safety_checker_idx_concepts = [[6749], [16630], [5937], [272, 279, 266], [11478], [19847], [817, 333], [9554], [13015, 1366], [11478, 2533], [33228, 4750], [569, 30951], [665, 868], [44774], [8626, 44774], [11478, 37637], [1076, 3373]]
    @torch.no_grad()
    def __init__(self, device="cpu", safety_model_id = "CompVis/stable-diffusion-safety-checker", cpu_offloading=False):
        self.device = device
        if cpu_offloading:
            if not is_accelerate_available():
                print("accelerate library is not installed! Unable to utilise CPU offloading!")
                cpu_offloading=False
            else:
                # NOTE: this only offloads parameters, but not buffers by default.
                from accelerate import cpu_offload
        self.safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)
        self.safety_processor = SafetyProcessor()
        globals()["_safety_embeddings"] = {"concepts":self.safety_checker.concept_embeds.cpu().clone(),"special":self.safety_checker.special_care_embeds.cpu().clone()}
        #safety_feature_extractor.to(device)
        if cpu_offloading:
            cpu_offload(self.safety_checker, OFFLOAD_EXEC_DEVICE, offload_buffers=False)
        else:
            self.safety_checker.to(self.device)

    def eval_concept_scores(self, cos_distances, thresholds, adjustment=0.0):
        results = {"scores":[None]*len(cos_distances), "detected_concepts":{}}
        for concept_idx in range(len(cos_distances)):
            concept_cos = cos_distances[concept_idx]
            concept_threshold = thresholds[concept_idx].item()
            results["scores"][concept_idx] = round(concept_cos - concept_threshold, 3)
            if results["scores"][concept_idx] > adjustment:
                results["detected_concepts"][concept_idx]=results["scores"][concept_idx]
        return results

    @torch.no_grad()
    def process_features(self, images):
        # adapted from the default safety checker code: Diffusers pipelines/stable_diffusion/safety_checker > StableDiffusionSafetyChecker
        safety_checker_input = self.safety_feature_extractor(images, return_tensors="pt").to(self.device)
        cosine_distance = lambda image_embeds, text_embeds: torch.mm(torch.nn.functional.normalize(image_embeds), torch.nn.functional.normalize(text_embeds).t())
        image_embeds = self.safety_checker.visual_projection(self.safety_checker.vision_model(safety_checker_input.pixel_values)[1])
        special_cos_dist = cosine_distance(image_embeds, self.safety_checker.special_care_embeds).cpu().float().numpy()
        cos_dist = cosine_distance(image_embeds, self.safety_checker.concept_embeds).cpu().float().numpy()
        results_per_image = []
        for i in range(image_embeds.shape[0]):
            special_care_results = self.eval_concept_scores(special_cos_dist[i], self.safety_checker.special_care_embeds_weights)
            nsfw_concepts_adj = 0.0 if len(special_care_results["detected_concepts"]) == 0 else 0.01
            nsfw_concept_results = self.eval_concept_scores(cos_dist[i], self.safety_checker.concept_embeds_weights, nsfw_concepts_adj)
            rating_level = (0 if len(nsfw_concept_results["detected_concepts"]) == 0 else 1) * (1 if len(special_care_results["detected_concepts"]) == 0 else 2)
            results_per_image.append({"special_care":special_care_results,"nsfw_concepts":nsfw_concept_results,"nsfw":rating_level})
        return results_per_image

    @torch.no_grad()
    def check_safety(self, images, level_override=None):
        eval_results = self.process_features(images)
        assert len(images) == len(eval_results)
        if any([x["nsfw"]>0 for x in eval_results]):
            detected_nsfw_concepts = [{**item["nsfw_concepts"]["detected_concepts"], **{-1-k:x for k,x in item["special_care"]["detected_concepts"].items()}} for item in eval_results]
            detected_concepts_repr = [str(item).replace(" ","").replace(":0.",":.") for item in detected_nsfw_concepts]
            tqdm.write(f"Potential NSFW content detected! ({len([x['nsfw'] for x in eval_results if x['nsfw']>0])} instances, levels {[x['nsfw'] for x in eval_results]}, labels {detected_concepts_repr})")
        for image,result in zip(images, eval_results):
            # Due to potential detection reliability issues with using CLIPSeg directly on "bad concept" labels, the standard safety checker is first employed to detect label presence.
            # Then, CLIPSeg is utilised to generate a blur mask for any areas most likely to contain detected labels, regardless of their absolute detection values.
            flagged_labels = result["nsfw_concepts"]["detected_concepts"]
            if len(flagged_labels) > 0:
                detected_labels = [SafetyChecker._safety_checker_idx_concepts[idx] for idx in flagged_labels] # safety_checker_idx_concepts
                self.safety_processor.process_image(image, result["nsfw"], detected_labels, level_override=level_override)
        return any([x["nsfw"]>0 for x in eval_results])
    def __call__(self, *args, **kwargs):
        # backwards compatiblity, replacing the former get_safety_checker(*) with SafetyChecker(*)
        return self.check_safety(*args, **kwargs)


# return LAB colorspace array for image
def image_to_correction_target(img):
    return cv2.cvtColor(np.asarray(img.copy()), cv2.COLOR_RGB2LAB)

def process_cycle_image(img:Image.Image, rotation:int=0, rotation_center:tuple=None, translation:tuple=(0,0), zoom:int=0, resample_sharpen:float=1.2, width:int=512, height:int=512, perform_histogram_correction:bool=True, correction_target:np.ndarray=None):
    from skimage import exposure
    background = img.copy()
    img = img.convert("RGBA")
    img = img.rotate(rotation, Image.Resampling.BILINEAR, expand=False, center=rotation_center, translate=translation)
    # zoom after rotating, not before (reduce background space created by translations/rotations)
    img = ImageOps.crop(img, zoom)
    img = img.resize((width,height), resample=Image.Resampling.LANCZOS)
    background.paste(img, (0, 0), img)
    img = background.convert("RGB")

    # boost sharpness for input into next cycle to avoid having the image become softer over time.
    if rotation !=0 or zoom !=0:
        enhancer_sharpen = ImageEnhance.Sharpness(img)
        img = enhancer_sharpen.enhance(resample_sharpen)

    if perform_histogram_correction and correction_target is not None:
        img = Image.fromarray(cv2.cvtColor(exposure.match_histograms(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2LAB), correction_target, channel_axis=2), cv2.COLOR_LAB2RGB).astype("uint8")).convert("RGB")

    return img

# write latent tensor to string for PNG metadata saving
def tensor_to_encoded_string(obj):
    bytes_io = BytesIO()
    torch.save(obj, bytes_io)
    # ascii85 encode raw binary data, ascii encode resulting bytes to string
    bytestring = base64.a85encode(bytes_io.getvalue()).decode("ascii")
    return bytestring

# load latent tensor (before VAE encode) from png metadata
# for grid images, index will select the correct image. For single images, only one index is available
def load_latents_from_image(path, index=0):
    try:
        pil_image = Image.open(path)
        if not hasattr(pil_image, "text"):
            return None
        elif not "latents" in pil_image.text:
            return None
        else:
            bytestring = pil_image.text["latents"]
            # ascii decode string back to a85 bytes, then a85 decode bytes back to raw binary data
            bytes_io = BytesIO(base64.a85decode(bytestring.encode("ascii")))
            params = torch.load(bytes_io)
            if len(params) <= index:
                print(f"Index {index} out of range: Tensor only contains {len(params)} items.")
                return None
            # original shape before VAE encode is tensor[image_index][...], recreate shape with selected index.
            return torch.stack([params[index]])
    except:
        print_exc()
        return None

# load any stored data from png metadata
def load_metadata_from_image(path):
    try:
        pil_image = Image.open(path)
        if not hasattr(pil_image, "text"):
            return None
        else:
            return pil_image.text
    except:
        print_exc()
        return None

# interpolate between two latents (if multiple images are contained in a latent, only the first one is used)
# returns latent sequence, which can be passed into VAE to retrieve image sequence.
@torch.no_grad()
def interpolate_latents(latent_a, latent_b, steps=1, overextend=0, granular_mode=False):
    _a = latent_a[0]
    _b = latent_b[0]
    if not granular_mode:
        a = _a + (_a-(_a+_b)/2)*overextend
        b = _b + (_b-(_a+_b)/2)*overextend
        # split interval [0..1] among step count
        size_per_step = 1/steps
        # as range(n) will iterate every item from 0 to n-1, 'a' (starting tensor of interpolation) is included. Attach ending tensor 'b' to the end.
        return torch.stack([torch.lerp(a,b, torch.full_like(a,size_per_step*step)) for step in tqdm(range(steps), position=0, desc="Interpolating")] + [b])
    else:
        assert _a.shape == _b.shape
        thresholds = torch.rand_like(_a) # uniformly distributed in [0,1[.
        # Each value in the latent tensor will switch from _a to _b when the merge progress is above the random threshold. Could be step/(steps-1), if we were not adding _b to the end.
        return torch.stack([_a*((thresholds>step/steps)*1.0) + _b*((thresholds<=step/steps)*1.0) for step in tqdm(range(steps), position=0, desc="Interpolating")] + [_b])

# accepts list of list of image. every sublist is one animation.
def save_animations(images_with_frames):
    TIMESTAMP = datetime.today().strftime('%Y%m%d%H%M%S')
    image_basepath = os.path.join(ANIMATIONS_DIR, TIMESTAMP)
    for (frames, image_index) in zip(images_with_frames, range(len(images_with_frames))):
        initial_image = frames[0]
        try:
            initial_image.save(fp=image_basepath+f"{image_index}_a.webp", format='webp', append_images=frames[1:], save_all=True, duration=65, minimize_size=True, loop=1)
            #initial_image.save(fp=image_basepath+f"{image_index}.gif", format='gif', append_images=frames[1:], save_all=True, duration=65) # GIF files take up lots of space.
            grid = image_autogrid(frames)
            grid_format = "webp" if all([s <= 16383 for s in grid.size]) else "jpeg" # webp supports a maximum of 16383 in any dimension. Since PIL does not (yet) support AVIF, fall back to JPEG.
            grid.save(fp=image_basepath+f"{image_index}_grid.{grid_format}", format=grid_format, minimize_size=True)
        except:
            print_exc()
        try:
            print("The following OpenCV codec error can be ignored: ", end="")
            video_writer = cv2.VideoWriter(f"{image_basepath}{image_index}.webm", cv2.VideoWriter_fourcc(*'VP90'), 15, initial_image.size)
            print()
            for frame in tqdm(frames, desc="Encoding webm video from frames!"):
                video_writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
            video_writer.release()
        except:
            print_exc()
    print(f"Saved files with prefix {image_basepath}")
    return image_basepath

def tensor_to_pil_compress_dynamic_range(image:torch.Tensor):
    image = (getattr(image, "sample", image) / 2 + 0.5)#.clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1)
    #print([(torch.min(im),torch.max(im))for im in image])
    image = torch.stack([im - torch.min(im)for im in image])
    image = torch.stack([im/torch.max(im)for im in image])
    image = image.clamp(0,1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def tensor_to_pil(image:torch.Tensor):
    image = (getattr(image, "sample", image) / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def create_interpolation(a, b, steps, vae, overextend=0.5):
    al = load_latents_from_image(a)
    bl = load_latents_from_image(b)
    with torch.no_grad():
        interpolated = interpolate_latents(al,bl,steps,overextend)
    print("decoding...")
    with torch.no_grad():
        # processing images in target device one-by-one saves VRAM.
        image_sequence = tensor_to_pil(torch.stack([vae.decode(torch.stack([item.to(DIFFUSION_DEVICE)]) * (1 / 0.18215))[0][0] for item in tqdm(interpolated, position=0, desc="Decoding latents")]))
    save_animations([image_sequence])

def re_encode(filepath, vae):
    latent = load_latents_from_image(filepath)
    with torch.no_grad():
        image = tensor_to_pil(vae.decode(latent.to(DIFFUSION_DEVICE) * (1 / 0.18215))[0])
    return image

class Placeholder():
    pass

placeholder = Placeholder()
class QuickGenerator():
    # can be used externally without requiring an instance
    def cleanup(devices):
        gc.collect()
        if "cuda" in devices:
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
    def __init__(self,tokenizer,text_encoder,unet,vae,IO_DEVICE,UNET_DEVICE,rate_nsfw,use_half_latents):
        QuickGenerator.cleanup([IO_DEVICE,UNET_DEVICE])
        self._tokenizer=tokenizer
        self._text_encoder=text_encoder
        self._unet=unet
        self._vae=vae
        self._IO_DEVICE=IO_DEVICE
        self._UNET_DEVICE=UNET_DEVICE
        self._rate_nsfw=rate_nsfw
        self._use_half_latents=use_half_latents
        self.generate_exec = generate_segmented(tokenizer=self._tokenizer,text_encoder=self._text_encoder,unet=self._unet,vae=self._vae,IO_DEVICE=self._IO_DEVICE,UNET_DEVICE=self._UNET_DEVICE,rate_nsfw=self._rate_nsfw,half_precision_latents=self._use_half_latents)

        # default config for re-initialisation
        self.default_config = {
            "sample_count":1,
            "width":512,
            "height":512,
            "steps":35,
            "guidance_scale":9,
            "seed":None,
            "sched_name":"mdpms",
            "ddim_eta":0.0,
            "eta_seed":None,
            "strength":0.75,
            "animate":False,
            "sequential_samples":False,
            "save_latents":True,
            "display_with_cv2":True,
            "attention_slicing":None,
            "animate_pred_diff":True,
            "gs_scheduler":None,
            "encoder_level_negative_prompts":False,
            "clip_skip_layers":0,
            "static_length":None,
            "mix_mode_concat":False,
            "controlnet":None,
            "controlnet_strength":1.0,
            "controlnet_preprocessor":None,
            "controlnet_strength_scheduler":None,
            "safety_processing_level":3,
            "guidance_rescale":0.66,
            "second_pass_resize":1,
            "second_pass_steps":50,
            "second_pass_use_controlnet":False,
        }
        # pre-set attributes as placeholders, ensuring that there are no missing attributes
        for attribute_name in self.default_config:
            setattr(self, attribute_name, self.default_config[attribute_name])
        # properly initialise config
        self.init_config()

    # (re)-initialise settings
    def init_config(self):
        self.configure(**self.default_config)

    # use Placeholder as the default value when the setting should not be changed. None is a valid value for some settings.
    def configure(self,
            sample_count:int=placeholder,
            width:int=placeholder,
            height:int=placeholder,
            steps:int=placeholder,
            guidance_scale:float=placeholder,
            seed:int=placeholder,
            sched_name:str=placeholder,
            ddim_eta:float=placeholder,
            eta_seed:int=placeholder,
            strength:float=placeholder,
            animate:bool=placeholder,
            sequential_samples:bool=placeholder,
            save_latents:bool=placeholder,
            display_with_cv2:bool=placeholder,
            attention_slicing:bool=placeholder,
            animate_pred_diff:bool=placeholder,
            gs_scheduler:str=placeholder,
            encoder_level_negative_prompts:bool=placeholder,
            clip_skip_layers:int=placeholder,
            static_length:int=placeholder,
            mix_mode_concat:bool=placeholder,
            controlnet:ControlNetModel=placeholder,
            controlnet_strength:float=placeholder,
            controlnet_preprocessor:str=placeholder,
            controlnet_strength_scheduler:str=placeholder,
            safety_processing_level:int=placeholder,
            guidance_rescale:float=placeholder,
            second_pass_resize:float=placeholder,
            second_pass_steps:int=placeholder,
            second_pass_use_controlnet:bool=placeholder,
        ):
        settings = locals()
        # if no additional processing is performed on an option, automate setting it on self:
        unmodified_options = [
            "guidance_scale","seed","ddim_eta","eta_seed","animate","sequential_samples","save_latents","display_with_cv2","animate_pred_diff","encoder_level_negative_prompts","clip_skip_layers","static_length","mix_mode_concat",
            "controlnet","controlnet_strength","controlnet_preprocessor","controlnet_strength_scheduler","guidance_rescale","second_pass_resize","second_pass_steps","second_pass_use_controlnet",
        ]
        for option in unmodified_options:
            if not isinstance(settings[option], Placeholder):
                setattr(self, option, settings[option])

        if not isinstance(sample_count, Placeholder):
            self.sample_count = max(1,sample_count)
        if not isinstance(width, Placeholder):
            self.width = int(width/8)*8
        if not isinstance(height, Placeholder):
            self.height = int(height/8)*8
        if not isinstance(steps, Placeholder):
            self.steps = max(1,steps)
        if not isinstance(strength, Placeholder):
            strength = strength if strength < 1.0 else 1.0
            strength = strength if strength > 0.0 else 0.0
            self.strength = strength
        if not isinstance(sched_name, Placeholder):
            if sched_name in IMPLEMENTED_SCHEDULERS:
                self.sched_name = sched_name
            else:
                print(f"unknown scheduler requested: '{sched_name}' options are: {IMPLEMENTED_SCHEDULERS}")
        if not isinstance(gs_scheduler, Placeholder):
            if gs_scheduler in IMPLEMENTED_GS_SCHEDULES:
                self.gs_scheduler = gs_scheduler
            else:
                print(f"unknown gs schedule requested: '{gs_scheduler}' options are {IMPLEMENTED_GS_SCHEDULES}")
        if not isinstance(safety_processing_level, Placeholder):
            self.safety_processing_level = safety_processing_level
            if not isinstance(self._rate_nsfw, SafetyChecker):
                print(f"WARNING: Safety processing level can not be switched to {safety_processing_level} as requested! No SafetyChecker is loaded, checking may be disabled.")
            else:
                self._rate_nsfw.safety_processor.set_processing_level(safety_processing_level)


        if not isinstance(attention_slicing, Placeholder) and not (attention_slicing==self.attention_slicing):
            # None disables slicing if parameter is False
            slice_size = None
            # 'True' / 0 -> use diffusers recommended 'automatic' value for "a good trade-off between speed and memory"
            # use bool instance check and bool value: otherwise (int)1==True -> True would be caught!
            use_recommended_slicing = (isinstance(attention_slicing, int) and attention_slicing <= 0) or (isinstance(attention_slicing, bool) and attention_slicing)

            if isinstance(self._unet.config.attention_head_dim, int):
                if use_recommended_slicing:
                    slice_size = self._unet.config.attention_head_dim//2
                # int -> use as slice size
                elif isinstance(attention_slicing, int):
                    slice_size = attention_slicing
                # False / None / unknown type will disable slicing with default value of None.
            elif isinstance(self._unet.config.attention_head_dim, list):
                if attention_slicing == 1:
                    # 1 is a valid setting.
                    pass
                # if any slicing is requested (not None, not False) with an attention list, pick the smallest size (diffusers pipeline implementation)
                elif use_recommended_slicing or (attention_slicing is not None and not (isinstance(attention_slicing,bool) and not attention_slicing)):
                    attention_slicing = min(self._unet.config.attention_head_dim)

            self.attention_slicing = attention_slicing
            self._unet.set_attention_slice(slice_size)
            if slice_size is not None:
                self._vae.enable_slicing()
            else:
                self._vae.disable_slicing()


    @torch.no_grad()
    def one_generation(self, prompt:str=None, init_image:Image.Image=None, save_images:bool=True, controlnet_input:Image.Image=None):
        # keep this conversion instead of only having a default value of "" to permit None as an input
        prompt = prompt if prompt is not None else ""
        prompts = [x.strip() for x in prompt.split("||")]*self.sample_count

        if init_image not in [None, []]:
            if not isinstance(init_image, list):
                init_image = [init_image]
            prepared_init_image = []
            for item in init_image:
                item = item.convert("RGB")
                if item.size != (self.width,self.height):
                    # if lanczos resampling is required, apply a small amount of post-sharpening
                    item = item.resize((self.width, self.height), resample=Image.Resampling.LANCZOS)
                    enhancer_sharpen = ImageEnhance.Sharpness(item)
                    item = enhancer_sharpen.enhance(1.15)
                prepared_init_image.append(item)
            init_image = prepared_init_image

        QuickGenerator.cleanup([self._IO_DEVICE,self._UNET_DEVICE])
        out, SUPPLEMENTARY = self.generate_exec(
            prompts,
            width=self.width,
            height=self.height,
            steps=self.steps,
            gs=self.guidance_scale,
            seed=self.seed,
            sched_name=self.sched_name,
            eta=self.ddim_eta,
            eta_seed=self.eta_seed,
            init_image=init_image,
            img_strength=self.strength,
            animate=self.animate,
            save_latents=self.save_latents,
            sequential=self.sequential_samples,
            gs_schedule=self.gs_scheduler,
            animate_pred_diff=self.animate_pred_diff,
            encoder_level_negative_prompts=self.encoder_level_negative_prompts,
            clip_skip_layers=self.clip_skip_layers,
            static_length=self.static_length,
            mix_mode_concat=self.mix_mode_concat,
            controlnet=self.controlnet,
            controlnet_strength=self.controlnet_strength,
            controlnet_preprocessor=self.controlnet_preprocessor,
            controlnet_input=controlnet_input,
            controlnet_strength_scheduler=self.controlnet_strength_scheduler,
            guidance_rescale=self.guidance_rescale,
            second_pass_resize=self.second_pass_resize,
            second_pass_steps=self.second_pass_steps,
            second_pass_use_controlnet=self.second_pass_use_controlnet,
        )
        argdict = SUPPLEMENTARY["io"]
        final_latent = SUPPLEMENTARY['latent']['final_latent']
        PIL_animation_frames = argdict.pop("image_sequence")
        # reduce prompt representation: only output one prompt instance for a larger batch of the same prompt
        argdict["text_readback"] = collapse_representation(argdict.pop("text_readback"), n=3)
        argdict["remaining_token_count"] = collapse_representation(argdict["remaining_token_count"], n=3)
        argdict["weights"] = collapse_representation(argdict.pop("weights"), n=3)
        argdict["clip_skip_layers"] = collapse_representation(argdict.pop("clip_skip_layers"),n=3)
        argdict["tokens"] = collapse_representation(argdict.pop("tokens"),n=3)
        if argdict["width"] == 512:
            argdict.pop("width")
        if argdict["height"] == 512:
            argdict.pop("height")
        if not argdict["nsfw"]:
            argdict.pop("nsfw")
        if not ("ddim" in argdict["sched_name"] and argdict["eta"] > 0):
            argdict.pop("eta")
            argdict.pop("eta_seed")
        if argdict["unet_model"] == argdict["vae_model"]:
            argdict.pop("vae_model")
            argdict["model"] = argdict.pop("unet_model")
        if argdict["gs_sched"] is None:
            argdict.pop("gs_sched")
        if argdict["encoder_level_negative"] is False:
            argdict.pop("encoder_level_negative")
        if argdict["clip_skip_layers"] == "0":
            argdict.pop("clip_skip_layers")
        if argdict["LoRA"] is None:
            argdict.pop("LoRA")
        if argdict["controlnet"] is None or argdict["controlnet_scheduler"] is None:
            argdict.pop("controlnet_scheduler")
        if argdict["controlnet"] is None:
            argdict.pop("controlnet")
        if argdict["static_length"] is None:
            argdict.pop("static_length")
            argdict.pop("remaining_token_count")
        if not argdict["concat"]:
            argdict.pop("concat")
        if argdict.get("sequential", False) and len(out) < 2:
            argdict.pop("sequential")
        full_image, full_metadata, metadata_items = save_output(prompts, out, argdict, save_images, final_latent, self.display_with_cv2)
        if self.animate:
            # init frames by image list
            frames_by_image = []
            # one sublist per image
            for _ in range(len(PIL_animation_frames[0])):
                frames_by_image.append([])
            for images_of_step in PIL_animation_frames:
                for (image, image_sublist_index) in zip(images_of_step, range(len(images_of_step))):
                    frames_by_image[image_sublist_index].append(image)
            save_animations(frames_by_image)
        return out,SUPPLEMENTARY, (full_image, full_metadata, metadata_items)

    @torch.no_grad()
    def perform_image_cycling(self,
            in_prompt:str="", img_cycles:int=100, save_individual:bool=False, save_final:bool=True,
            init_image:Image.Image=None, text2img:bool=False, color_correct:bool=False, color_target_image:Image.Image=None,
            zoom:int=0,rotate:int=0,center=None,translate=None,sharpen:int=1.0, controlnet_input:Image.Image=None,
            select_next_image_in_batch:bool=False
        ):
        # sequence prompts across all iterations
        prompts = [p.strip() for p in in_prompt.split("||")]
        iter_per_prompt = (img_cycles / (len(prompts)-1)) if text2img and (len(prompts) > 1) else img_cycles / len(prompts)

        # set/modify prompt weights with a given factor
        def process_prompt_weights(prompt,scale_mult):
            prompt_segments = re.split(";(-[\.0-9]+|[\.0-9]*\+?);", prompt)
            if prompt_segments[-1] == "":
                # if last prompt in sequence is terminated by a weight, remove trailing empty segment created by splitting.
                prompt_segments = prompt_segments[:-1]
            if len(prompt_segments) % 2 == 1:
                # if the final/only segment does not have a trailing factor, give it the factor 1
                prompt_segments.append("1")
            resulting_prompt = ""
            for i, item in enumerate(prompt_segments):
                # text-prompt segments are appended as-is. Trailing weight segments are appended inside ;; with the scale factor applied. For empty weights, use base weight of 1
                resulting_prompt += item if i % 2 == 0 else f";{(1 if item in ['','+'] else int(item[:-1]) if item.endswith('+') else int(item))*scale_mult:.5}{'+' if item.endswith('+') else ''};"
            return resulting_prompt

        def index_prompt(i):
            prompt_idx = int(i/iter_per_prompt)
            progress_in_idx = i % iter_per_prompt
            # interpolate through the prompts in the text encoding space, using prompt mixing. Set weights from the frame count between the prompts
            if not prompt_idx+1 in range(len(prompts)):
                return prompts[prompt_idx]
            else:
                # apply scale factors to both the previous and upcoming text prompts, then concatenate them. Both will already terminate with trailing weights.
                if PROMPT_INTERPOLATION_CROSSOVER_POINT != 0.5:
                    if progress_in_idx <= iter_per_prompt/2:
                        progress_in_first_half = progress_in_idx/(iter_per_prompt/2)
                        first_segment_scale = (1-progress_in_first_half) + progress_in_first_half*PROMPT_INTERPOLATION_CROSSOVER_POINT # scale first segment from 1 to crossover
                        second_segment_scale = progress_in_first_half*PROMPT_INTERPOLATION_CROSSOVER_POINT # scale second segment from 0 to crossover
                    else:
                        progress_in_second_half = (progress_in_idx-iter_per_prompt/2)/(iter_per_prompt/2)
                        first_segment_scale = (1-progress_in_second_half)*PROMPT_INTERPOLATION_CROSSOVER_POINT # scale first segment from crossover to 0
                        second_segment_scale = (1-progress_in_second_half)*PROMPT_INTERPOLATION_CROSSOVER_POINT + progress_in_second_half # scale second segment from crossover to 1
                else:
                    first_segment_scale = (iter_per_prompt-progress_in_idx)/iter_per_prompt
                    second_segment_scale = progress_in_idx/iter_per_prompt
                first_segment = process_prompt_weights(prompts[prompt_idx], first_segment_scale)
                second_segment = process_prompt_weights(prompts[prompt_idx+1], second_segment_scale)
                # switching the segment with the higher magnitude to the front has not been found to make a difference thus far.
                return f"{first_segment}{second_segment}" #if progress_in_idx <= iter_per_prompt/2 else f"{second_segment}{first_segment}"

        correction_target = None
        # first frame is the initial image, if it exists.
        frames = []
        try:
            cycle_bar = tqdm(range(img_cycles), position=0, desc="Image cycle")
            for i in cycle_bar:
                # if correction should be applied, the correction target array is not present, and a source image to generate the array from is present, create it.
                if color_correct and correction_target is None and (init_image is not None or color_target_image is not None):
                    correction_target = image_to_correction_target(color_target_image if color_target_image is not None else init_image)
                # if text-to-image should be used, set input image for cycle to None
                init_image = init_image if not text2img else None
                next_prompt = index_prompt(i)
                cycle_bar.set_description(next_prompt if len(next_prompt) <= 64 else f'{next_prompt[:28]} (...) {next_prompt[-28:]}')
                out, SUPPLEMENTARY, _ = self.one_generation(next_prompt, init_image, save_images=save_individual, controlnet_input=controlnet_input)
                # when using a fixed seed for img2img cycling, advance it by one.
                self.seed = self.seed if self.seed is None or text2img else self.seed + 1
                if "cuda" in [IO_DEVICE, DIFFUSION_DEVICE]:
                    gc.collect()
                    torch.cuda.empty_cache()

                if select_next_image_in_batch:
                    selected_idx=0
                    if self.display_with_cv2 and not text2img and len(out) > 1:
                        tqdm.write("Select next image to continue the sequence")
                        @dataclass
                        class ClickContainer():
                            click_x=-1
                            click_y=-1
                        container = ClickContainer()
                        cv2.waitKey(1)
                        def callback_handler(event,x,y,flags,param):
                            if event == cv2.EVENT_LBUTTONUP:
                                container.click_x,container.click_y = x,y
                        cv2.setMouseCallback(CV2_TITLE, callback_handler)
                        while cv2.getWindowProperty(CV2_TITLE, cv2.WND_PROP_VISIBLE):
                            cv2.waitKey(1)
                            if -1 not in [container.click_x,container.click_y]:
                                break
                        selected_col = math.trunc(container.click_x / (self.width+GRID_IMAGE_SEPARATION))
                        selected_row = math.trunc(container.click_y / (self.height+GRID_IMAGE_SEPARATION))
                        (_xpos, _ypos, window_width, window_height) = cv2.getWindowImageRect(CV2_TITLE)
                        selected_idx = int(selected_col + selected_row*round(window_width/(self.width+1),0))
                        selected_idx = selected_idx if selected_idx in range(len(out)) else 0
                        tqdm.write(f"Selected: index {selected_idx} in [0,{len(out)-1}]")
                    next_frame = out[selected_idx]
                    frames.append(next_frame.copy())
                else:
                    next_frame = out
                    if len(frames) == 0 and init_image is not None:
                        # init with an autogrid of the correct size: amount of actual copies must match output batch size.
                        frames = [image_autogrid([init_image.resize((out[0].size), resample=Image.Resampling.LANCZOS)]*len(out))]
                    frames.append(image_autogrid(out))
                if not text2img:
                    # skip processing image for next cycle if it will not be used.
                    if isinstance(next_frame, list):
                        init_image = [process_cycle_image(item, rotate, center, translate, zoom, sharpen, self.width, self.height, color_correct, correction_target) for item in next_frame]
                    else:
                        init_image = process_cycle_image(next_frame, rotate, center, translate, zoom, sharpen, self.width, self.height, color_correct, correction_target)

        except KeyboardInterrupt:
            # when an interrupt is received, cancel cycles and attempt to save any already generated images.
            pass

        # if images are not saved on every step, save the final image separately
        argdict = SUPPLEMENTARY["io"]
        final_latent = SUPPLEMENTARY['latent']['final_latent']
        full_image, full_metadata, metadata_items = save_output(in_prompt, out, argdict, (not save_individual) and save_final, final_latent, False)
        if save_final:
            animation_files_basepath = save_animations([frames])
        return out,SUPPLEMENTARY, (full_image, full_metadata, metadata_items)

def controlnet_aux_extractor(detector_model, *args, **kwargs):
    try:
        import controlnet_aux as ca
    except ImportError:
        print(f"unable to import detector for {detector_model} from 'controlnet_aux'. Please install it via 'pip install controlnet_aux'")
        return None
    detector_map = {"detect_pose":ca.OpenposeDetector, "detect_mlsd":ca.MLSDdetector, "detect_hed":ca.HEDdetector} # , "detect_midas":ca.MidasDetector} # seems to not be in pypi yet, hold until next release
    if not detector_model in detector_map:
        print(f"Unknown aux detector model: {detector_model}!")
        return None
    Detector = detector_map[detector_model]
    model = Detector.from_pretrained("lllyasviel/ControlNet")
    results = model(*args, **kwargs)
    del model
    gc.collect()
    return results

class SafetyCheckerProcessingLevel():
    def __init__(self, quantile:float=0.9, threshold_range:float=0.66, remove:bool=False):
        self.quantile = quantile
        self.threshold_range = threshold_range
        self.remove = remove
class SafetyProcessorProcessingLevels():
    remove = SafetyCheckerProcessingLevel(0.0, 1.0, True) # default safety checker action
    full_blur = SafetyCheckerProcessingLevel(0.0, 1.0, False) # alternative to 'remove'
    strong_masked_blur = SafetyCheckerProcessingLevel(0.75, 0.66, False)
    masked_blur = SafetyCheckerProcessingLevel(0.95, 0.66, False)
    ignore = SafetyCheckerProcessingLevel(1.0, -1, False)
    def get_level(level_name:str):
        return getattr(SafetyProcessorProcessingLevels, level_name, SafetyProcessorProcessingLevels.strong_masked_blur)

class SafetyProcessor():
    def __init__(self, processing_level:int=3, clipseg_model:str=SAFETY_PROCESSOR_CLIPSEG_MODEL, device=IO_DEVICE):
        self.clipseg_model = clipseg_model
        self.device=device
        self.segmentation_preprocessor = None
        self.segmentation_model = None
        self.set_processing_level(processing_level)
    def set_processing_level(self, level:int=3):
        self._processing_level = level
        if level == 0:
            print("WARNING: Safety processing level set to 0, disabling processing entirely! Setting this is generally NEVER recommended.")
        elif level <= 2:
            print("WARNING: Safety processing level set below 3. This will permit any potentially NSFW outputs as long as no special labels are detected!")
        levels = {
            0:["ignore", "ignore"],
            1:["ignore", "masked_blur"],
            2:["ignore", "strong_masked_blur"],
            3:["masked_blur", "strong_masked_blur"],
            4:["masked_blur", "full_blur"],
            5:["strong_masked_blur", "full_blur"],
            6:["full_blur", "full_blur"], # alternative to "remove"
            7:["remove", "remove"], # default behavior of a StableDiffusionPipeline
            -1:["masked_blur", "masked_blur"], # debug
            -2:["strong_masked_blur", "strong_masked_blur"], # debug
        }
        # if the requested level does not exist, use one of the maximum levels (6)
        self.processing_by_rating = [SafetyProcessorProcessingLevels.get_level(level_name) for level_name in levels.get(self._processing_level, 6)]
        # only load the models when they actually get utilised.
        #if self._processing_level != 0 and None in [self.segmentation_preprocessor, self.segmentation_model]:
        #    self.load_missing_model()
    def load_missing_model(self):
        if self.segmentation_preprocessor is None:
            self.segmentation_preprocessor = CLIPSegProcessor.from_pretrained(self.clipseg_model)
        if self.segmentation_model is None:
            # do not move the model to the device yet - keep it in CPU memory while it remains unused. It will temporarily be moved to the device when processing.
            self.segmentation_model = CLIPSegForImageSegmentation.from_pretrained(self.clipseg_model)

    def process_image(self, img:Image.Image, nsfw_rating:int, detected_labels:list, full_blur_fallback:bool=False, level_override:int=None):
        if level_override is not None:
            # temporarily adjust level, produce an output, and adjust the level back.
            previous_level = self._processing_level
            self.set_processing_level(level_override)
            results = self.process_image(img=img,nsfw_rating=nsfw_rating,detected_labels=detected_labels,full_blur_fallback=full_blur_fallback)
            self.set_processing_level(previous_level)
            return results
        if nsfw_rating == 0:
            return img
        processing_level = self.processing_by_rating[nsfw_rating-1] # rating 1 is at index 0, rating 2 at index 1

        if processing_level.threshold_range < 0 or processing_level.quantile > 1:
            return img
        elif processing_level.remove:
            img.paste(GRID_IMAGE_BACKGROUND_COLOR, [0,0,img.size[0],img.size[1]])
            return img
        elif processing_level.threshold_range >= 1.0 or processing_level.quantile <= 0 or full_blur_fallback:
            blurred = cv2.resize(cv2.resize(np.asarray(img), [16,16]), img.size, interpolation=cv2.INTER_AREA)
            blurred = Image.fromarray(blurred.round().astype("uint8"))
            img.paste(blurred)
            return blurred
        else:
            if len(detected_labels) == 0:
                print("WARNING: Safety processor was requested to perform masked blur, but no labels were provided! Performing full blur.")
                return self.process_image(img=img, nsfw_rating=nsfw_rating, detected_labels=detected_labels, full_blur_fallback=True, level_override=level_override)
            segment_predictions = self.get_segmentation(img, detected_labels)
            blurred = self.process_masked_blur(img, segment_predictions, processing_level)
            img.paste(blurred)
            return blurred

    @torch.no_grad()
    def get_segmentation(self, img:Image.Image, prompts:list):
        try:
            if None in [self.segmentation_preprocessor, self.segmentation_model]:
                self.load_missing_model()
            self.segmentation_model.to(self.device)
            prediction_maps = []
            for prompt_item in prompts:
                # run sequentially to minimize memory requirement on multiple prompts
                if isinstance(prompt_item, list) and all([isinstance(x, int) for x in prompt_item]):
                    # if the prompt is a list of int, treat it as a token id list. Place it between the start and end tokens generated by an empty prompt, and re-pack batch dim.
                    model_inputs = self.segmentation_preprocessor(text=[""], images=[img], return_tensors="pt")
                    model_inputs["input_ids"] = torch.LongTensor([[model_inputs["input_ids"][0][0], *prompt_item, model_inputs["input_ids"][0][1]]])
                    model_inputs["attention_mask"] = torch.LongTensor([[1]*len(model_inputs["input_ids"][0])]) # Amended attention mask length
                else:
                    # the input data consists of very short, individually batched prompts. Skip padding to max length.
                    model_inputs = self.segmentation_preprocessor(text=[prompt_item], images=[img], return_tensors="pt") # , padding="max_length")
                model_inputs = {k:v.to(self.device) for k,v in model_inputs.items()}
                # apply sigmoid activation
                raw_prediction = self.segmentation_model(**model_inputs).logits.cpu()
                prediction_maps.append(torch.sigmoid(raw_prediction))
            self.segmentation_model.to("cpu")
            return prediction_maps
        except RuntimeError as e:
            if 'out of memory' in str(e) and self.device !="cpu":
                self.device="cpu"
                return self.get_segmentation(img,prompts)
            else:
                raise e
    def process_masked_blur(self, img:Image.Image, location_predictions:torch.Tensor, processing_level:SafetyCheckerProcessingLevel):
        # accumulate every relevant mask
        mask = None
        for pred in location_predictions:
            data = pred.cpu().float().numpy()
            threshold = np.quantile(data,processing_level.quantile)*processing_level.threshold_range # set threshold: e.g. within 33% (absolute value) of the 98th percentile
            threshold = min(threshold, 0.18) # set a lower bound of (absolute) confidence for the threshold.
            new_mask = (data>threshold) * 1.0 # mask any pixels within the threshold
            mask = new_mask if mask is None else mask+new_mask
        # cumulatively stack masks. Expand mask borders with a blur, resulting values will be thresholded to 1. (Expansion equivalent to the kernel radius, on unscaled segmentation outputs)
        mask = cv2.GaussianBlur(mask, (33,33), 5)
        mask = (mask>0.01) * 1.0

        blursize = (min(round(img.size[0]/24+2), 24), min(round(img.size[1]/24+2), 24)) # global size for pixelation blur of masked areas # (16,16)
        #mask = cv2.resize(mask, dsize=blursize, interpolation=cv2.INTER_LANCZOS4) # decreasing mask resolution could be used to create more coarse, less detailed mask
        blur_mask = cv2.resize(mask, dsize=img.size, interpolation=cv2.INTER_AREA) # resize mask to image size, use a smooth/blurry interpolation to achieve some edge roll-off and avoid artifacting
        blur_mask = cv2.blur(blur_mask, (9,9)) # apply blur after scaling to improve edge roll-off
        if SAFETY_PROCESSOR_DISPLAY_MASK_CV2:
            show_image(Image.fromarray((blur_mask*255).round().astype("uint8")), "SafetyProcessor mask")
        # If something other than local pixelation blur is desired, this image data (blurred_data) should be replaced! (e.g. replacing with a flat color, smooth blur, image noise, etc.)
        blurred_data = cv2.resize(cv2.resize(np.asarray(img), blursize, interpolation=cv2.INTER_AREA), img.size, interpolation=cv2.INTER_AREA) # apply pixelation blur to source image (global)
        blur_mask = np.stack([blur_mask]*3, axis=-1) # stack mask to 3 values per pixel
        resulting = blur_mask*blurred_data + np.asarray(img)*(1-blur_mask) # merge the original and blurred images based on the mask
        return Image.fromarray(resulting.round().astype("uint8"))

def choose_int(upper_bound:int):
    if upper_bound <= 1:
        return 0
    else:
        try:
            val = int(input(f"Select: [0-{upper_bound-1}] > "))
            assert val in range(upper_bound)
            return val
        except:
            return 0

# unpack object from potential (nested) single-element list(s)
def unpack_encapsulated_lists(x):
    while isinstance(x, list) and len(x) == 1:
        x = x[0]
    return x

# TODO: Remake this function at some point
# create a string from a list, while removing duplicate string items in a row. Returns "[]" for empty lists. n>1 recursively re-applies the function.
def collapse_representation(item_list,keep_lowest_level=False,n=1):
    return str(do_collapse_representation(item_list,keep_lowest_level,n))
def do_collapse_representation(item_list, keep_lowest_level=False, n=1):
    if n <= 0:
        return item_list
    item_list = unpack_encapsulated_lists(item_list)
    collapsed = []
    if isinstance(item_list, list) and ((not keep_lowest_level) or (all([isinstance(x,list) for x in item_list]))):
        for item in item_list:
            # drop sequential repetitions: only append the next item if it differs from the last item to be added
            item = do_collapse_representation(item,keep_lowest_level,n-1)
            if len(collapsed) == 0 or collapsed[-1] != item:
                collapsed.append(item)
    else:
        return item_list
    # unpack a second time, turning a potential list of only a single item into the item itself
    collapsed = unpack_encapsulated_lists(collapsed)
    return collapsed

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Cancelled.")
