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
from transformers import models as transfomers_models
from diffusers import models as diffusers_models
from transformers import CLIPTextModel, CLIPTokenizer, AutoFeatureExtractor
from diffusers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, IPNDMScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler, HeunDiscreteScheduler
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.utils import is_accelerate_available
import argparse
from datetime import datetime
from PIL.PngImagePlugin import PngInfo
import codecs
import cv2
import numpy as np
from traceback import print_exc
from PIL import Image, ImageOps, ImageEnhance
from tqdm.auto import tqdm
from time import time
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
import inspect
from io import BytesIO
from pathlib import Path
import re
from tokens import get_huggingface_token
import uuid

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
IO_DEVICE = "cpu"
# execution device when offloading enabled. Required as a global for now.
OFFLOAD_EXEC_DEVICE = "cuda"
# display current image in img2img cycle mode via cv2.
IMAGE_CYCLE_DISPLAY_CV2 = True
# global flag signifying if a v_prediction model is in use.
V_PREDICTION_MODEL = False
# will be read and overridden from the text encoder layer count on model load. Inclusive range limit. 12 for ViT-L/14 (SD1.x) or 23 for ViT-H (SD2.x)
MAX_CLIP_SKIP = 12

OUTPUT_DIR_BASE = "outputs"
OUTPUTS_DIR = f"{OUTPUT_DIR_BASE}/generated"
ANIMATIONS_DIR = f"{OUTPUT_DIR_BASE}/animate"
INDIVIDUAL_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "individual")
UNPUB_DIR = os.path.join(OUTPUTS_DIR, "unpub")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(INDIVIDUAL_OUTPUTS_DIR, exist_ok=True)
os.makedirs(UNPUB_DIR, exist_ok=True)
os.makedirs(ANIMATIONS_DIR, exist_ok=True)

IMPLEMENTED_SCHEDULERS = ["lms", "pndm", "ddim", "euler", "euler_ancestral", "mdpms", "sdpms", "kdpm2", "kdpm2_ancestral", "heun"] # ipndm not supported
V_PREDICTION_SCHEDULERS = IMPLEMENTED_SCHEDULERS # ["euler", "ddim", "mdpms", "euler_ancestral", "pndm", "lms", "sdpms"]
IMPLEMENTED_GS_SCHEDULES = [None, "sin", "cos", "isin", "icos", "fsin", "anneal5", "rand", "frand"]

# sd2.0 default negative prompt
DEFAULT_NEGATIVE_PROMPT = "" # e.g. dreambot SD2.0 default : "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft"

# THIS WIL AUTOMATICALLY RESET. DO NOT ADD CUSTOM VALUES HERE. substitutions/shortcuts in prompts. Will be filled with substitions for multi-token custom embeddings.
PROMPT_SUBST = {}
# Custom combinations of "shortname":"substitution string", could be added here.
CUSTOM_PROMPT_SUBST = {
    "<negative>":"ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, blurry, bad anatomy, blurred, watermark, grainy, signature, cut off, draft",
}

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
    parser.add_argument("--io-device", type=str, default="cpu", help="Device for running text encoding and VAE decoding. Keep on CPU for reduced VRAM load.", dest="io_device")
    parser.add_argument("--animate", action="store_true", help="save animation of generation process. Very slow unless --io_device is set to \"cuda\"", dest="animate")
    parser.add_argument("-in", "--interpolate", nargs=2, type=str, help="Two image paths for generating an interpolation animation", default=None, dest='interpolation_targets')
    parser.add_argument("-inx", "--interpolate_extend", type=float, help="Interpolate beyond the specified images in the latent space by the given factor. Disabled with 0. Unlikely to provide useful results.", default=0, dest='interpolation_extend')
    parser.add_argument("--no-check-nsfw", action="store_true", help="NSFW check will only print a warning and add a metadata label if enabled. This flag disables NSFW check entirely to speed up generation.", dest="no_check")
    parser.add_argument("-pf", "--prompts-file", type=str, help="Path of file containing prompts. One line per prompt.", default=None, dest='prompts_file')
    parser.add_argument("-ic", "--image-cycles", type=int, help="Repetition count when using image2image. Will interpolate between multiple prompts when || is used.", default=0, dest='img_cycles')
    parser.add_argument("-cni", "--cycle-no-save-individual", action="store_false", help="Disables saving of individual images in image cycle mode.", dest="image_cycle_save_individual")
    parser.add_argument("-iz", "--image-zoom", type=int, help="Amount of zoom (pixels cropped per side) for each subsequent img2img cycle", default=0, dest='img_zoom')
    parser.add_argument("-ir", "--image-rotate", type=int, help="Amount of rotation (counter-clockwise degrees) for each subsequent img2img cycle", default=0, dest='img_rotation')
    parser.add_argument("-it", "--image-translate", type=int, help="Amount of translation (x,y tuple in pixels) for each subsequent img2img cycle", default=None, nargs=2, dest='img_translation')
    parser.add_argument("-irc", "--image-rotation-center", type=int, help="Center of rotational axis when applying rotation (0,0 -> top left corner) Default is the image center", default=None, nargs=2, dest='img_center')
    parser.add_argument("-ics", "--image-cycle-sharpen", type=float, default=1.2, help="Sharpening factor applied when zooming and/or rotating. Reduces effect of constant image blurring due to resampling. Sharpens at >1.0, softens at <1.0", dest="cycle_sharpen")
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
    return parser.parse_args()

def main():
    global DIFFUSION_DEVICE, IO_DEVICE
    global model_id, models_local_dir
    global OUTPUT_DIR_BASE, OUTPUTS_DIR, INDIVIDUAL_OUTPUTS_DIR, UNPUB_DIR, ANIMATIONS_DIR
    global DEFAULT_NEGATIVE_PROMPT
    args = parse_args()
    if args.cpu_offloading:
        args.diffusion_device, args.io_device = OFFLOAD_EXEC_DEVICE, OFFLOAD_EXEC_DEVICE
    DIFFUSION_DEVICE = args.diffusion_device
    IO_DEVICE = args.io_device
    DEFAULT_NEGATIVE_PROMPT = args.default_negative

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

    generator = QuickGenerator(tokenizer,text_encoder,unet,vae,IO_DEVICE,DIFFUSION_DEVICE,rate_nsfw,args.half_latents)
    generator.configure(args.n_samples, args.width, args.height, args.steps, args.guidance_scale, args.seed, args.sched_name, args.ddim_eta, args.eta_seed, args.strength, args.animate, args.sequential_samples, True, True, args.attention_slicing, True, args.gs_schedule, args.negative_prompt_mixing, args.clip_skip_layers)

    init_image = None if args.init_img_path is None else Image.open(args.init_img_path).convert("RGB")

    if args.img_cycles > 0:
        generator.perform_image_cycling(args.prompt, args.img_cycles, args.image_cycle_save_individual, init_image, args.cycle_fresh, args.cycle_color_correction, None, args.img_zoom, args.img_rotation, args.img_center, args.img_translation, args.cycle_sharpen)
        exit()

    if args.prompts_file is not None:
        lines = [line.strip() for line in codecs.open(args.prompts_file, "r").readlines()]
        for line in tqdm(lines, position=0, desc="Input prompts"):
            generator.one_generation(line, init_image)
    elif args.prompt is not None:
        generator.one_generation(args.prompt, init_image)
    else:
        while True:
            prompt = input("Prompt> ").strip()
            try:
                generator.one_generation(prompt, init_image)
            except KeyboardInterrupt:
                pass
            except Exception as e:
                print_exc()

def load_models(half_precision=False, unet_only=False, cpu_offloading=False, vae_only=False):
    global using_local_unet, using_local_vae, PROMPT_SUBST, MAX_CLIP_SKIP
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
    tokenizer:transfomers_models.clip.tokenization_clip.CLIPTokenizer
    text_encoder:transfomers_models.clip.modeling_clip.CLIPTextModel
    unet:diffusers_models.unet_2d_condition.UNet2DConditionModel
    vae:diffusers_models.vae.AutoencoderKL

    unet_model_id = model_id
    vae_model_id = model_id
    text_encoder_id = "openai/clip-vit-large-patch14" # sd 1.x default
    tokenizer_id = "openai/clip-vit-large-patch14" # sd 1.x default
    use_auth_token_unet=get_huggingface_token()
    use_auth_token_vae=get_huggingface_token()

    default_local_model_files_per_dir = ["config.json", "diffusion_pytorch_model.bin"]
    if models_local_dir is not None and len(models_local_dir) > 0:
        unet_dir = os.path.join(models_local_dir, "unet")
        if all([os.path.exists(os.path.join(unet_dir, f)) for f in default_local_model_files_per_dir]):
            use_auth_token_unet=False
            print(f"Using local unet model files! ({models_local_dir})")
            unet_model_id = unet_dir
            using_local_unet = True
        else:
            print(f"Using unet '{unet_model_id}' (no local data present at {unet_dir})")
        vae_dir = os.path.join(models_local_dir, "vae")
        if all([os.path.exists(os.path.join(vae_dir, f)) for f in default_local_model_files_per_dir]):
            use_auth_token_vae=False
            print(f"Using local vae model files! ({models_local_dir})")
            vae_model_id = vae_dir
            using_local_vae = True
        else:
            print(f"Using vae '{vae_model_id}' (no local data present at {vae_dir})")
        text_encoder_dir = os.path.join(models_local_dir, "text_encoder")
        if all([os.path.exists(os.path.join(text_encoder_dir, f)) for f in ["config.json", "pytorch_model.bin"]]):
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

    if not vae_only:
        # Load the UNet model for generating the latents.
        if half_precision:
            unet = UNet2DConditionModel.from_pretrained(unet_model_id, subfolder="unet", use_auth_token=use_auth_token_unet, torch_dtype=torch.float16, revision="fp16")
        else:
            unet = UNet2DConditionModel.from_pretrained(unet_model_id, subfolder="unet", use_auth_token=use_auth_token_unet)

        if cpu_offloading:
            cpu_offload(unet, execution_device=OFFLOAD_EXEC_DEVICE, offload_buffers=True)

        if unet_only:
            return unet

    # Load the autoencoder model which will be used to decode the latents into image space.
    if False:
        vae = AutoencoderKL.from_pretrained(vae_model_id, subfolder="vae", use_auth_token=use_auth_token_vae, torch_dtype=torch.float16)
    else:
        vae = AutoencoderKL.from_pretrained(vae_model_id, subfolder="vae", use_auth_token=use_auth_token_vae)
    if vae_only:
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
    rate_nsfw = get_safety_checker(cpu_offloading=False)
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

    # text encoder should be offloaded after adding custom embeddings to ensure that every embedding is actually on the same device.
    if cpu_offloading:
        cpu_offload(vae, execution_device=OFFLOAD_EXEC_DEVICE, offload_buffers=True)
        cpu_offload(text_encoder, execution_device=OFFLOAD_EXEC_DEVICE, offload_buffers=True)

    return tokenizer, text_encoder, unet, vae, rate_nsfw

def generate_segmented(
        tokenizer:transfomers_models.clip.tokenization_clip.CLIPTokenizer,
        text_encoder:transfomers_models.clip.modeling_clip.CLIPTextModel,
        unet:diffusers_models.unet_2d_condition.UNet2DConditionModel,
        vae:diffusers_models.vae.AutoencoderKL,
        IO_DEVICE="cpu",UNET_DEVICE="cuda",rate_nsfw=(lambda x: False),half_precision_latents=False
    ):
    if not vae.device == torch.device("meta"):
        vae.to(IO_DEVICE)
    if not text_encoder.device == torch.device("meta"):
        text_encoder.to(IO_DEVICE)
    if not unet.device == torch.device("meta"):
        unet.to(UNET_DEVICE)
    UNET_DEVICE = UNET_DEVICE if UNET_DEVICE != "meta" else OFFLOAD_EXEC_DEVICE
    IO_DEVICE = IO_DEVICE if IO_DEVICE != "meta" else OFFLOAD_EXEC_DEVICE
    #if UNET_DEVICE == "meta" and unet.dtype == torch.float16:
    #    half_precision_latents = True

    @torch.no_grad()
    def perform_text_encode(prompt,clip_skip_layers=0):
        io_data = {}

        prompt = prompt.strip()
        clip_skip_prompt_override = re.search('\{cls[0-9]+\}$', prompt) # match only '{cls<int>}' at the end of the prompt
        if clip_skip_prompt_override is not None:
            clip_skip_layers = min(int(clip_skip_prompt_override[0][4:-1]), MAX_CLIP_SKIP) # take match, drop leading '{cls' and trailing '}'. Clamp to possible value limit
            # since this only matches on the end of the string, splitting multiple times on unfortunate prompts is not possible
            prompt = re.split('\{cls[0-9]+\}$', prompt)[0]
        io_data["clip_skip_layers"] = [clip_skip_layers]
        # set rerun layers *after* checking for skip override. this allows producing of 'special tensors' with a trailing clip skip override
        rerun_self_kwargs = {"clip_skip_layers":clip_skip_layers}

        # apply all prompt substitutions. Process longest substitution first, to avoid substring collisions.
        sorted_substitutions = sorted([(k,PROMPT_SUBST[k],uuid.uuid4()) for k in PROMPT_SUBST], key=lambda x: len(x[0]), reverse=True)
        # first, grab all substitutions and replace them by a temporary UUID. This avoids substring collisions between substitution values and subsequent (shorter) substitution keys.
        for (p,subst,identifier) in sorted_substitutions:
            prompt = prompt.replace(p, f"<{identifier}>")
        # then, replace UUID placeholders with substitution values. Substring collisions should not be an issue.
        for (p,subst,identifier) in sorted_substitutions:
            prompt = prompt.replace(f"<{identifier}>", subst)

        # text embeddings
        text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        num_truncated_tokens = text_input.num_truncated_tokens
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
        io_data["remaining_token_count"] = [(- item) if item>0 else (io_data["text_readback"][idx].count("<|endoftext|>") - 1) for (item,idx) in zip(num_truncated_tokens,range(len(num_truncated_tokens)))]
        # If there is more space when looking at SD2.0 padding (trailing '!'), output it instead.
        io_data["remaining_token_count"] = [max(remaining, trailing) for (remaining,trailing) in zip(io_data["remaining_token_count"], trailing_char_counts)]
        io_data["text_readback"] = [s.replace("<|endoftext|>","").replace("<|startoftext|>","") for s in io_data["text_readback"]]
        io_data["attention"] = text_input.attention_mask.cpu().numpy().tolist()
        # TODO: implement custom attention masks?
        if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
            attention_mask = text_input.attention_mask.to(IO_DEVICE)
        else:
            attention_mask = None
        if clip_skip_layers <= 0: # a value of 0 (index: [-1]) would yield the final layer output, equivalent to default behavior
            text_embeddings = text_encoder(text_input.input_ids.to(IO_DEVICE), attention_mask=attention_mask)[0]
        else:
            text_embeddings = text_encoder(text_input.input_ids.to(IO_DEVICE), attention_mask=attention_mask, output_hidden_states=True).hidden_states[-(clip_skip_layers+1)]
            text_embeddings = text_encoder.text_model.final_layer_norm(text_embeddings)

        if prompt == "<damaged_uncond>":
            uncond, _ = perform_text_encode("", **rerun_self_kwargs)
            for i in range(uncond.shape[1]):
                # create some tensor damage: mix half of the vector elements with a random element of the same vector.
                idx = torch.randperm(uncond.shape[2])
                for j in range(uncond.shape[2]):
                    if torch.rand(1) > 0.5:
                        uncond[:,i,j] = (uncond[:,i,j]*2 + uncond[:,i,idx[j]]) /3
            text_embeddings=uncond
        elif prompt == "<alpha_dropout_uncond>":
            uncond, _ = perform_text_encode("", **rerun_self_kwargs)
            text_embeddings = torch.nn.functional.alpha_dropout(uncond, p=0.5)
        elif prompt == "<all_starts>":
            uncond, _ = perform_text_encode("", **rerun_self_kwargs)
            for i in range(1,uncond.shape[1]):
                uncond[:,i] = uncond[:,0]
            text_embeddings = uncond
        elif prompt == "<all_ends>":
            uncond, _ = perform_text_encode("", **rerun_self_kwargs)
            for i in range(0,uncond.shape[1]-1):
                uncond[:,i] = uncond[:,-1]
            text_embeddings = uncond
        elif prompt == "<all_start_end>":
            uncond, _ = perform_text_encode("", **rerun_self_kwargs)
            for i in range(1,uncond.shape[1]-1):
                uncond[:,i] = uncond[:,0] if i%2==0 else uncond[:,-1]
            text_embeddings = uncond
        return text_embeddings, io_data

    @torch.no_grad()
    def encode_prompt(prompt, encoder_level_negative_prompts=False, clip_skip_layers=0):
        perform_encode_kwargs = {"clip_skip_layers":clip_skip_layers}
        io_data = {"text_readback":[], "remaining_token_count":[], "attention":[], "clip_skip_layers":[]}
        embeddings_list_uncond = []
        embeddings_list_prompt = []
        for raw_item in prompt:
            # create sequence of substrings followed by weight multipliers: substr,w,substr,w,... (w may be empty string - use default: 1, last trailing substr may be empty string)
            prompt_segments = re.split(";(-[\.0-9]+|[\.0-9]*);", raw_item)
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
                new_io_data = {"text_readback":[], "remaining_token_count":[], "attention":[], "clip_skip_layers":[]}
                for i, segment in enumerate(prompt_segments):
                    if i % 2 == 0:
                        next_encoding, next_io_data = perform_text_encode(segment, **perform_encode_kwargs)
                        for key in next_io_data:
                            # glue together the io_data sublist of all the prompts being mixed
                            new_io_data[key] += next_io_data[key]
                    else:
                        # get multiplier. 1 if no multiplier is present.
                        multiplier = 1 if segment == "" else float(segment)
                        new_io_data["text_readback"].append(f";{multiplier};")
                        # add either to positive, or to negative encodings & multipliers
                        is_negative_multiplier = multiplier < 0
                        if is_negative_multiplier:
                            multiplier = abs(multiplier)
                            multiplier_sum_negative += multiplier
                            if encodings_sum_negative is None:
                                # if sum is empty, write first item.
                                encodings_sum_negative = next_encoding*multiplier
                            else:
                                # add new encodings to sum based on their multiplier
                                encodings_sum_negative += next_encoding*multiplier
                        else:
                            multiplier_sum_positive += multiplier
                            if encodings_sum_positive is None:
                                # if sum is empty, write first item.
                                encodings_sum_positive = next_encoding*multiplier
                            else:
                                # add new encodings to sum based on their multiplier
                                encodings_sum_positive += next_encoding*multiplier

                if encoder_level_negative_prompts:
                    # old mixing mode. leaves uncond embeddings untouched, mixes negative prompts directly into the prompt embedding itself.
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
            

    @torch.no_grad()
    def generate_segmented_exec(
            prompt=["art"], width=512, height=512, steps=50, gs=7.5, seed=None, sched_name="pndm", eta=0.0, eta_seed=None, high_beta=False,
            animate=False, init_image=None, img_strength=0.5, save_latents=False, gs_schedule=None, animate_pred_diff=True,
            encoder_level_negative_prompts=False, clip_skip_layers=0
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
                "attention" : [],
                "unet_model" : f"{models_local_dir} (local)" if using_local_unet else model_id,
                "vae_model" : f"{models_local_dir} (local)" if using_local_vae else model_id,
                "encoder_level_negative" : encoder_level_negative_prompts,
                "clip_skip_layers" : [clip_skip_layers], # will be overridden by encode_prompt
            },
            "latent" : {
                "final_latent" : None,
                "latent_sequence" : [],
            },
        }

        # truncate incorrect input dimensions to a multiple of 64
        if isinstance(prompt, str):
            prompt = [prompt]
        height = int(height/64.0)*64
        width = int(width/64.0)*64
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

        text_embeddings, batch_size, io_data = encode_prompt(prompt, encoder_level_negative_prompts, clip_skip_layers)
        for key in io_data:
            SUPPLEMENTARY["io"][key] = io_data[key]

        # schedulers
        if high_beta:
            # scheduler default
            beta_start = 0.0001
            beta_end = 0.02
        else:
            # stablediffusion default
            beta_start = 0.00085
            beta_end = 0.012
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

        else:
            raise ValueError(f"Requested unknown scheduler: {sched_name}")
        # set timesteps. Also offset, as pipeline does this
        accepts_offset = "offset" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if accepts_offset:
            scheduler_offset=1
            scheduler.set_timesteps(steps, offset=1)
        else:
            scheduler_offset=0
            scheduler.set_timesteps(steps)

        starting_timestep = 0
        # initial noise latents
        if init_image is None:
            latents = torch.randn((batch_size, unet.in_channels, height // 8, width // 8), device=IO_DEVICE, generator=generator_unet)
            if half_precision_latents:
                latents = latents.to(dtype=torch.float16)
            latents = latents.to(UNET_DEVICE)
        else:
            SUPPLEMENTARY["io"]["strength"] = img_strength
            init_image = init_image.resize((width,height), resample=Image.Resampling.LANCZOS)
            image_processing = np.array(init_image).astype(np.float32) / 255.0
            image_processing = image_processing[None].transpose(0, 3, 1, 2)
            init_image_tensor = 2.0 * torch.from_numpy(image_processing) - 1.0
            init_latents = vae.encode(init_image_tensor.to(IO_DEVICE)).latent_dist.sample()
            # apply inverse scaling
            init_latents = 0.18215 * init_latents
            init_latents = torch.cat([init_latents] * batch_size)
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

        # denoising loop!
        progress_bar = tqdm([x for x in enumerate(scheduler.timesteps[starting_timestep:])], position=1)
        for i, t in progress_bar:
            if animate and not animate_pred_diff:
                SUPPLEMENTARY["latent"]["latent_sequence"].append(latents.clone().cpu())
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            if hasattr(scheduler, "scale_model_input"):
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            with torch.no_grad():
                cast_device = UNET_DEVICE #if UNET_DEVICE != "meta" else OFFLOAD_EXEC_DEVICE
                noise_pred = unet(latent_model_input.to(device=cast_device, dtype=unet.dtype), t.to(device=cast_device, dtype=unet.dtype), encoder_hidden_states=text_embeddings.to(device=cast_device, dtype=unet.dtype))["sample"]
            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            progress_factor = i/len(scheduler.timesteps[starting_timestep:])
            gs_mult = get_gs_mult(gs_schedule, progress_factor)
            progress_bar.set_description(f"gs={gs*gs_mult:.3f}")
            noise_pred = noise_pred_uncond + gs * (noise_pred_text - noise_pred_uncond) * gs_mult

            if animate and animate_pred_diff:
                SUPPLEMENTARY["latent"]["latent_sequence"].append((latents-noise_pred).clone().cpu())
            del noise_pred_uncond, noise_pred_text
            # compute the previous noisy sample x_t -> x_t-1
            if isinstance(scheduler, LMSDiscreteScheduler):
                #latents = scheduler.step(noise_pred, i, latents)["prev_sample"]
                latents = scheduler.step(noise_pred, t, latents)["prev_sample"]
            elif isinstance(scheduler, DDIMScheduler):
                latents = scheduler.step(noise_pred, t, latents, eta=eta, generator=generator_eta)["prev_sample"]
            else:
                latents = scheduler.step(noise_pred, t, latents)["prev_sample"]

            del noise_pred

        # free up some now unused memory before attempting VAE decode!
        del latent_model_input, scheduler, text_embeddings
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
        image = vae_decode_with_failover(vae, latents)

        # latents are decoded and copied (if requested) by now.
        del latents

        def to_pil(image, should_rate_nsfw:bool=True):
            image = (image.sample / 2 + 0.5).clamp(0, 1)
            #image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            if should_rate_nsfw:
                has_nsfw = rate_nsfw(image)
            else:
                has_nsfw = False
            images = (image * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]
            return pil_images, has_nsfw

        pil_images, has_nsfw = to_pil(image)
        SUPPLEMENTARY["io"]["nsfw"] = has_nsfw
        if animate:
            # swap places between UNET and VAE to speed up large batch decode. then swap back.
            unet.to(IO_DEVICE)
            vae.to(DIFFUSION_DEVICE)
            torch.cuda.synchronize()
            with torch.no_grad():
                # process items one-by-one to avoid overfilling VRAM with a batch containing all items at once.
                SUPPLEMENTARY["io"]["image_sequence"] = [to_pil(vae_decode_with_failover(vae,(item*(1 / 0.18215)).to(DIFFUSION_DEVICE,vae.dtype)), False)[0] for item in tqdm(SUPPLEMENTARY["latent"]["latent_sequence"], position=0, desc="Decoding animation latents")]
            torch.cuda.synchronize()
            vae.to(IO_DEVICE)
            unet.to(DIFFUSION_DEVICE)
        SUPPLEMENTARY["io"]["time"] = time() - START_TIME

        return pil_images, SUPPLEMENTARY

    def cleanup_memory():
        gc.collect()
        if "cuda" in [IO_DEVICE, DIFFUSION_DEVICE]:
            torch.cuda.empty_cache()

    # can optionally move the unet out of the way to make space
    @torch.no_grad()
    def vae_decode_with_failover(vae, latents:torch.Tensor):
        try:
            image_out = vae.decode(latents)
            return image_out
        except RuntimeError as e:
            if 'out of memory' in str(e):
                if vae.device != "cpu":
                    return vae_decode_cpu(vae, latents)
                else:
                    raise RuntimeError(f"VAE decode ran out of memory, with VAE already on CPU: {e}")
            else:
                # there was some RuntimeError, but it wasn't OOM.
                raise e

    @torch.no_grad()
    def vae_decode_cpu(vae, latents:torch.Tensor):
        tqdm.write("Decode moved to CPU.")
        cleanup_memory()
        try:
            prev_device = vae.device
            vae = vae.to(device="cpu")
            latents = latents.to(device="cpu")
            image_out = vae.decode(latents)
            vae = vae.to(device=prev_device)
        except NotImplementedError as e:
            if "Cannot copy out of meta tensor" in str(e):
                # vae is offloaded through accelerate. use a temporary stand-in vae.
                new_vae = load_models(vae_only=True)
                new_vae = new_vae.to(device="cpu")
                latents = latents.to(device="cpu")
                image_out = new_vae.decode(latents)
                del new_vae
                cleanup_memory()
        return image_out

    @torch.no_grad()
    def generate_segmented_wrapper(
            prompt=["art"], width=512, height=512, steps=50, gs=7.5, seed=None, sched_name="pndm", eta=0.0, eta_seed=None,
            animate=False, init_image=None, img_strength=0.5, save_latents=False, sequential=False, gs_schedule=None, animate_pred_diff=True,
            encoder_level_negative_prompts=False, clip_skip_layers=0
        ):
        exec_args = {
            "width":width,"height":height,"steps":steps,"gs":gs,"seed":seed,"sched_name":sched_name,"eta":eta,"eta_seed":eta_seed,"high_beta":False,
            "animate":animate,"init_image":init_image,"img_strength":img_strength,"save_latents":save_latents,"gs_schedule":gs_schedule,"animate_pred_diff":animate_pred_diff,
            "encoder_level_negative_prompts":encoder_level_negative_prompts,"clip_skip_layers":clip_skip_layers,
        }
        if not sequential:
            try:
                return generate_segmented_exec(prompt=prompt, **exec_args)
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
                    new_out, new_supplementary = generate_segmented_exec(prompt=prompt_item, **exec_args)
                    # reassemble list of outputs
                    out += new_out
                    seeds.append(new_supplementary['io']['seed'])
                    eta_seeds.append(new_supplementary['io']['eta_seed'])
                    if SUPPLEMENTARY is None:
                        SUPPLEMENTARY = new_supplementary
                    else:
                        if save_latents:
                            # reassemble stack tensor of final latents
                            SUPPLEMENTARY['latent']['final_latent'] = torch.stack([x for x in SUPPLEMENTARY['latent']['final_latent']] + [new_supplementary['latent']['final_latent'][0]])
                        # reassemble other supplementary return data
                        SUPPLEMENTARY['io']['text_readback'] += new_supplementary['io']['text_readback']
                        SUPPLEMENTARY['io']['remaining_token_count'] += new_supplementary['io']['remaining_token_count']
                        SUPPLEMENTARY['io']['time'] += new_supplementary['io']['time']
                        SUPPLEMENTARY['io']['attention'] += new_supplementary['io']['attention']
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
def save_output(p, imgs, argdict, perform_save=True, latents=None, display=False):
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

    grid_latent_string = tensor_to_encoded_string(latents)
    # keep original shape: tensor[image_index][...], by creating tensor with only one index
    if multiple_images:
        if latents is None:
            item_latent_strings = [""]*len(imgs)
        else:
            item_latent_strings = [tensor_to_encoded_string(torch.stack([x])) for x in latents]

    TIMESTAMP = datetime.today().strftime('%Y%m%d%H%M%S')
    prompts_for_filename, pm = cleanup_str(p)
    prompts_for_metadata = pm.replace("\n","\\n")

    argdict_str = str(argdict).replace('\n', '\\n')

    item_metadata_list = []
    if multiple_images:
        for (new_img, i) in zip(imgs, range(len(imgs))):
            _, item_pm = cleanup_str(p[i])
            item_prompt_for_metadata = item_pm.replace("\n","\\n")
            item_metadata = PngInfo()
            item_metadata.add_text("prompt", f"{item_prompt_for_metadata}\n{argdict_str}")
            item_metadata.add_text("index", str(i))
            item_metadata.add_text("latents", item_latent_strings[i])
            filepath_noext = os.path.join(INDIVIDUAL_OUTPUTS_DIR, f"{TIMESTAMP}_{prompts_for_filename}_{i}")
            if perform_save:
                new_img.save(filepath_noext+".png", pnginfo=item_metadata)
            item_metadata_list.append(item_metadata)

    base_filename = f"{TIMESTAMP}_{prompts_for_filename}"
    filepath_noext = os.path.join(OUTPUTS_DIR, f'{base_filename}')

    metadata = PngInfo()
    metadata.add_text("prompt", f"{prompts_for_metadata}\n{argdict_str}")
    metadata.add_text("latents", grid_latent_string)

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

def show_image(img):
    title="Output"
    try:
        cv2.imshow(title, cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
        cv2.waitKey(5)
    except:
        pass

# function to create one image containing all input images in a grid.
# currently not intended for images of differing sizes.
def image_autogrid(imgs, fixed_rows=None) -> Image:
    GRID_IMAGE_SEPARATION = 1
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
    w += GRID_IMAGE_SEPARATION
    h += GRID_IMAGE_SEPARATION
    # remove one image separation size from the overall size (no added padding after the final row/col)
    grid = Image.new('RGB', size=(cols*w-GRID_IMAGE_SEPARATION, rows*h-GRID_IMAGE_SEPARATION))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

@torch.no_grad()
def get_safety_checker(device="cpu", safety_model_id = "CompVis/stable-diffusion-safety-checker", cpu_offloading=False):
    if cpu_offloading:
        if not is_accelerate_available():
            print("accelerate library is not installed! Unable to utilise CPU offloading!")
            cpu_offloading=False
        else:
            # NOTE: this only offloads parameters, but not buffers by default.
            from accelerate import cpu_offload
    safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)
    #safety_feature_extractor.to(device)
    if cpu_offloading:
        cpu_offload(safety_checker, OFFLOAD_EXEC_DEVICE, offload_buffers=False)
    else:
        safety_checker.to(device)
    def numpy_to_pil(images):
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images
    def check_safety(x_image_in):
        x_image = x_image_in.copy()
        safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
        safety_checker_input.to(device)
        x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
        if True in has_nsfw_concept:
            tqdm.write("Warning: potential NSFW content detected!")
        assert x_checked_image.shape[0] == len(has_nsfw_concept)
        return True in has_nsfw_concept

    return check_safety

# return LAB colorspace array for image
def image_to_correction_target(img):
    return cv2.cvtColor(np.asarray(img.copy()), cv2.COLOR_RGB2LAB)

def process_cycle_image(img:Image, rotation:int=0, rotation_center:tuple=None, translation:tuple=(0,0), zoom:int=0, resample_sharpen:float=1.2, width:int=512, height:int=512, perform_histogram_correction:bool=True, correction_target:np.ndarray=None):
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
        print(print_exc())
        return None

# interpolate between two latents (if multiple images are contained in a latent, only the first one is used)
# returns latent sequence, which can be passed into VAE to retrieve image sequence.
@torch.no_grad()
def interpolate_latents(latent_a, latent_b, steps=1, overextend=0):
    _a = latent_a[0]
    _b = latent_b[0]
    a = _a + (_a-(_a+_b)/2)*overextend
    b = _b + (_b-(_a+_b)/2)*overextend
    # split interval [0..1] among step count
    size_per_step = 1/steps
    # as range(n) will iterate every item from 0 to n-1, 'a' (starting tensor of interpolation) is included. Attach ending tensor 'b' to the end.
    return torch.stack([torch.lerp(a,b, torch.full_like(a,size_per_step*step)) for step in tqdm(range(steps), position=0, desc="Interpolating")] + [b])

# accepts list of list of image. every sublist is one animation.
def save_animations(images_with_frames):
    TIMESTAMP = datetime.today().strftime('%Y%m%d%H%M%S')
    image_basepath = os.path.join(ANIMATIONS_DIR, TIMESTAMP)
    for (frames, image_index) in zip(images_with_frames, range(len(images_with_frames))):
        initial_image = frames[0]
        try:
            initial_image.save(fp=image_basepath+f"{image_index}_a.webp", format='webp', append_images=frames[1:], save_all=True, duration=65, minimize_size=True, loop=1)
            #initial_image.save(fp=image_basepath+f"{image_index}.gif", format='gif', append_images=frames[1:], save_all=True, duration=65) # GIF files take up lots of space.
            image_autogrid(frames).save(fp=image_basepath+f"{image_index}_grid.webp", format='webp', minimize_size=True)
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

def create_interpolation(a, b, steps, vae, overextend=0.5):
    al = load_latents_from_image(a)
    bl = load_latents_from_image(b)
    with torch.no_grad():
        interpolated = interpolate_latents(al,bl,steps,overextend)
    def to_pil(image:torch.Tensor):
        image = (getattr(image, "sample", image) / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images
    print("decoding...")
    with torch.no_grad():
        # processing images in target device one-by-one saves VRAM.
        image_sequence = to_pil(torch.stack([vae.decode(torch.stack([item.to(DIFFUSION_DEVICE)]) * (1 / 0.18215))[0][0] for item in tqdm(interpolated, position=0, desc="Decoding latents")]))
    save_animations([image_sequence])

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
        ):
        settings = locals()
        # if no additional processing is performed on an option, automate setting it on self:
        unmodified_options = ["guidance_scale","seed","ddim_eta","eta_seed","animate","sequential_samples","save_latents","display_with_cv2","animate_pred_diff","encoder_level_negative_prompts","clip_skip_layers"]
        for option in unmodified_options:
            if not isinstance(settings[option], Placeholder):
                setattr(self, option, settings[option])

        if not isinstance(sample_count, Placeholder):
            self.sample_count = max(1,sample_count)
        if not isinstance(width, Placeholder):
            self.width = int(width/64)*64
        if not isinstance(height, Placeholder):
            self.height = int(height/64)*64
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


    @torch.no_grad()
    def one_generation(self, prompt:str=None, init_image:Image=None, save_images:bool=True):
        # keep this conversion instead of only having a default value of "" to permit None as an input
        prompt = prompt if prompt is not None else ""
        prompts = [x.strip() for x in prompt.split("||")]*self.sample_count

        if init_image is not None:
            init_image = init_image.convert("RGB")
            if init_image.size != (self.width,self.height):
                # if lanczos resampling is required, apply a small amount of post-sharpening
                init_image = init_image.resize((self.width, self.height), resample=Image.Resampling.LANCZOS)
                enhancer_sharpen = ImageEnhance.Sharpness(init_image)
                init_image = enhancer_sharpen.enhance(1.15)

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
        )
        argdict = SUPPLEMENTARY["io"]
        final_latent = SUPPLEMENTARY['latent']['final_latent']
        PIL_animation_frames = argdict.pop("image_sequence")
        # reduce prompt representation: only output one prompt instance for a larger batch of the same prompt
        argdict["text_readback"] = collapse_representation(argdict.pop("text_readback"))
        argdict["remaining_token_count"] = collapse_representation(argdict["remaining_token_count"])
        argdict["attention"] = collapse_representation(argdict["attention"], keep_lowest_level=True, n=2)
        argdict["clip_skip_layers"] = collapse_representation(argdict.pop("clip_skip_layers"))
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
            in_prompt:str="", img_cycles:int=100, save_individual:bool=False,
            init_image:Image=None, text2img:bool=False, color_correct:bool=False, color_target_image:Image=None,
            zoom:int=0,rotate:int=0,center=None,translate=None,sharpen:int=1.2,
        ):
        # sequence prompts across all iterations
        prompts = [p.strip() for p in in_prompt.split("||")]
        iter_per_prompt = (img_cycles / (len(prompts)-1)) if text2img and (len(prompts) > 1) else img_cycles / len(prompts)

        # set/modify prompt weights with a given factor
        def process_prompt_weights(prompt,scale_mult):
            prompt_segments = re.split(";(-[\.0-9]+|[\.0-9]*);", prompt)
            if prompt_segments[-1] == "":
                # if last prompt in sequence is terminated by a weight, remove trailing empty segment created by splitting.
                prompt_segments = prompt_segments[:-1]
            if len(prompt_segments) % 2 == 1:
                # if the final/only segment does not have a trailing factor, give it the factor 1
                prompt_segments.append("1")
            resulting_prompt = ""
            for i, item in enumerate(prompt_segments):
                # text-prompt segments are appended as-is. Trailing weight segments are appended inside ;; with the scale factor applied. For empty weights, use base weight of 1
                resulting_prompt += item if i % 2 == 0 else f";{(1 if item=='' else int(item))*scale_mult};"
            return resulting_prompt

        def index_prompt(i):
            prompt_idx = int(i/iter_per_prompt)
            progress_in_idx = i % iter_per_prompt
            # interpolate through the prompts in the text encoding space, using prompt mixing. Set weights from the frame count between the prompts
            if not prompt_idx+1 in range(len(prompts)):
                return prompts[prompt_idx]
            else:
                # apply scale factors to both the previous and upcoming text prompts, then concatenate them. Both will already terminate with trailing weights.
                return f"{process_prompt_weights(prompts[prompt_idx], iter_per_prompt-progress_in_idx)}{process_prompt_weights(prompts[prompt_idx+1], progress_in_idx)}"

        correction_target = None
        # first frame is the initial image, if it exists.
        frames = [] if init_image is None else [init_image.resize((self.width, self.height), resample=Image.Resampling.LANCZOS)]
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
                out, SUPPLEMENTARY, _ = self.one_generation(next_prompt, init_image, save_images=save_individual)
                if "cuda" in [IO_DEVICE, DIFFUSION_DEVICE]:
                    gc.collect()
                    torch.cuda.empty_cache()

                next_frame = out[0]
                frames.append(next_frame.copy())
                if not text2img:
                    # skip processing image for next cycle if it will not be used.
                    init_image = process_cycle_image(next_frame, rotate, center, translate, zoom, sharpen, self.width, self.height, color_correct, correction_target)

        except KeyboardInterrupt:
            # when an interrupt is received, cancel cycles and attempt to save any already generated images.
            pass

        # if images are not saved on every step, save the final image separately
        argdict = SUPPLEMENTARY["io"]
        final_latent = SUPPLEMENTARY['latent']['final_latent']
        full_image, full_metadata, metadata_items = save_output(in_prompt, out, argdict, (not save_individual), final_latent, False)
        animation_files_basepath = save_animations([frames])


# unpack object from potential (nested) single-element list(s)
def unpack_encapsulated_lists(x):
    while isinstance(x, list) and len(x) == 1:
        x = x[0]
    return x

# create a string from a list, while removing duplicate string items in a row. Returns "[]" for empty lists. n>1 recursively re-applies the function.
def collapse_representation(item_list, keep_lowest_level=False, n=1):
    if n <= 0:
        return str(item_list)
    item_list = unpack_encapsulated_lists(item_list)
    collapsed = []
    if isinstance(item_list, list) and ((not keep_lowest_level) or (all([isinstance(x,list) for x in item_list]))):
        for item in item_list:
            # drop sequential repetitions: only append the next item if it differs from the last item to be added
            if len(collapsed) == 0 or collapsed[-1] != item:
                collapsed.append(item)
    else:
        return collapse_representation(item_list, keep_lowest_level, n-1)
    # unpack a second time, turning a potential list of only a single item into the item itself
    collapsed = unpack_encapsulated_lists(collapsed)
    return collapse_representation(collapsed, keep_lowest_level, n-1)

if __name__ == "__main__":
    main()
