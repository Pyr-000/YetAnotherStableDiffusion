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
from torch import autocast
from transformers import models as transfomers_models
from diffusers import models as diffusers_models
from transformers import CLIPTextModel, CLIPTokenizer, AutoFeatureExtractor
from diffusers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers import AutoencoderKL, UNet2DConditionModel
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

# for automated downloads via huggingface hub
model_id = "CompVis/stable-diffusion-v1-4"
# for manual model installs
models_local_dir = "models/stable-diffusion-v1-4"
# one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve, ort, mps, xla, lazy, vulkan, meta, hpu
# devices will be set by argparser if used. 
DIFFUSION_DEVICE = "cuda"
IO_DEVICE = "cpu"
# display current image in img2img cycle mode via cv2.
IMAGE_CYCLE_DISPLAY_CV2 = True

OUTPUTS_DIR = "outputs/generated"
ANIMATIONS_DIR = "outputs/animate"
INDIVIDUAL_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "individual")
UNPUB_DIR = os.path.join(OUTPUTS_DIR, "unpub")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(INDIVIDUAL_OUTPUTS_DIR, exist_ok=True)
os.makedirs(UNPUB_DIR, exist_ok=True)
os.makedirs(ANIMATIONS_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(type=str, nargs="?", default=None, help="text prompt for generation. Leave unset to enter prompt loop mode. Multiple prompts can be separated by ||.", dest="prompt")
    parser.add_argument("-ii", "--init-img", type=str, default=None, help="use img2img mode. path to the input image", dest="init_img_path")
    parser.add_argument("-st", "--strength", type=float, default=0.75, help="strength of initial image in img2img. 0.0 -> no new information, 1.0 -> only new information", dest="strength")
    parser.add_argument("-s","--steps", type=int, default=50, help="number of sampling steps", dest="steps")
    parser.add_argument("-sc","--scheduler", type=str, default="pndm", choices=["lms", "pndm", "ddim"], help="scheduler used when sampling in the diffusion loop", dest="sched_name")
    parser.add_argument("-e", "--ddim-eta", type=float, default=0.0, help="eta adds additional random noise when sampling, only applied on ddim sampler. 0 -> deterministic", dest="ddim_eta")
    parser.add_argument("-es","--ddim-eta-seed", type=int, default=None, help="secondary seed for the eta noise with ddim sampler and eta>0", dest="eta_seed")
    parser.add_argument("-H","--H", type=int, default=512, help="image height, in pixel space", dest="height")
    parser.add_argument("-W","--W", type=int, default=512, help="image width, in pixel space", dest="width")
    parser.add_argument("-n", "--n-samples", type=int, default=1, help="how many samples to produce for each given prompt. A.k.a. batch size", dest="n_samples")
    parser.add_argument("-cs", "--scale", type=float, default=7.5, help="(classifier free) guidance scale (higher values may increse adherence to prompt but decrease 'creativity')", dest="guidance_scale")
    parser.add_argument("-S","--seed", type=int, default=None, help="initial noise seed for reproducing/modifying outputs (None will select a random seed)", dest="seed")
    parser.add_argument("--unet-full", action='store_false', help="Run diffusion UNET at full precision (fp32). Default is half precision (fp16). Increases memory load.", dest="half")
    parser.add_argument("--latents-half", action='store_true', help="Generate half precision latents (fp16). Default is full precision latents (fp32), memory usage will only reduce by <1MB. Outputs will be slightly different.", dest="half_latents")
    parser.add_argument("--diff-device", type=str, default="cuda", help="Device for running diffusion process", dest="diffusion_device")
    parser.add_argument("--io-device", type=str, default="cpu", help="Device for running text encoding and VAE decoding. Keep on CPU for reduced VRAM load.", dest="io_device")
    parser.add_argument("--animate", action="store_true", help="save animation of generation process. Very slow unless --io_device is set to \"cuda\"", dest="animate")
    parser.add_argument("-in", "--interpolate", nargs=2, type=str, help="Two image paths for generating an interpolation animation", default=None, dest='interpolation_targets')
    parser.add_argument("--no-check-nsfw", action="store_true", help="NSFW check will only print a warning and add a metadata label if enabled. This flag disables NSFW check entirely to speed up generation.", dest="no_check")
    parser.add_argument("-pf", "--prompts-file", type=str, help="Path of file containing prompts. One line per prompt.", default=None, dest='prompts_file')
    parser.add_argument("-ic", "--image-cycles", type=int, help="Repetition count when using image2image", default=0, dest='img_cycles')
    parser.add_argument("-cni", "--cycle-no-save-individual", action="store_false", help="Disables saving of individual images in image cycle mode.", dest="image_cycle_save_individual")
    parser.add_argument("-iz", "--image-zoom", type=int, help="Amount of zoom (pixels cropped per side) for each subsequent img2img cycle", default=0, dest='img_zoom')
    parser.add_argument("-ir", "--image-rotate", type=int, help="Amount of rotation (counter-clockwise degrees) for each subsequent img2img cycle", default=0, dest='img_rotation')
    parser.add_argument("-it", "--image-translate", type=int, help="Amount of translation (x,y tuple in pixels) for each subsequent img2img cycle", default=None, nargs=2, dest='img_translation')
    parser.add_argument("-irc", "--image-rotation-center", type=int, help="Center of rotational axis when applying rotation (0,0 -> top left corner) Default is the image center", default=None, nargs=2, dest='img_center')
    parser.add_argument("-ics", "--image-cycle-sharpen", type=float, default=1.2, help="Sharpening factor applied when zooming and/or rotating. Reduces effect of constant image blurring due to resampling. Sharpens at >1.0, softens at <1.0", dest="cycle_sharpen")
    parser.add_argument("-icc", "--image-color-correction", action="store_true", help="When cycling images, keep lightness and colorization distribution the same as it was initially (LAB histogram cdf matching). Prevents 'magenta shifting' with multiple cycles.", dest="cycle_color_correction")
    return parser.parse_args()

def main():
    global DIFFUSION_DEVICE
    global IO_DEVICE
    args = parse_args()
    DIFFUSION_DEVICE = args.diffusion_device
    IO_DEVICE = args.io_device
    
    # load up models
    tokenizer, text_encoder, unet, vae, rate_nsfw = load_models(half_precision=args.half)
    if args.no_check:
        rate_nsfw = lambda x: False
    # create generate function using the loaded models

    # if interpolation is requested, run interpolation instead!
    if args.interpolation_targets is not None:
        # swap places between UNET and VAE to speed up large batch decode.
        unet = unet.to(IO_DEVICE)
        vae = vae.to(DIFFUSION_DEVICE)
        create_interpolation(args.interpolation_targets[0], args.interpolation_targets[1], args.steps, vae)
        exit()

    # fix values to multiples of 64 before init image resize in case of img2img cycling
    args.width = int(args.width/64)*64
    args.height = int(args.height/64)*64
    init_image = None
    if args.init_img_path is not None:
        init_image = Image.open(args.init_img_path).convert("RGB")
        if init_image.size != (args.width,args.height):
            # if lanczos resampling is required, apply a small amount of post-sharpening
            init_image = init_image.resize((args.width, args.height), resample=Image.Resampling.LANCZOS)
            enhancer_sharpen = ImageEnhance.Sharpness(init_image)
            init_image = enhancer_sharpen.enhance(1.15)

    args.strength = args.strength if args.strength < 1.0 else 1.0
    args.strength = args.strength if args.strength > 0.0 else 0.0
    generate_exec = generate_segmented(tokenizer=tokenizer,text_encoder=text_encoder,unet=unet,vae=vae,IO_DEVICE=IO_DEVICE,UNET_DEVICE=DIFFUSION_DEVICE,rate_nsfw=rate_nsfw,half_precision_latents=args.half_latents)

    def one_generation(prompt, init_image, display_with_cv2=False, save_latents=True, save_images=True):
        prompt = prompt if prompt is not None else ""
        prompts = [x.strip() for x in prompt.split("||")]
        out, SUPPLEMENTARY = generate_exec(prompts*args.n_samples, width=args.width, height=args.height, steps=args.steps, gs=args.guidance_scale, seed=args.seed, sched_name=args.sched_name, eta=args.ddim_eta, eta_seed=args.eta_seed, init_image=init_image, img_strength=args.strength, animate=args.animate, save_latents=save_latents)
        argdict = SUPPLEMENTARY["io"]
        final_latent = SUPPLEMENTARY['latent']['final_latent']
        PIL_animation_frames = argdict.pop("image_sequence")
        if argdict["width"] == 512:
            argdict.pop("width")
        if argdict["height"] == 512:
            argdict.pop("height")
        if not argdict["nsfw"]:
            argdict.pop("nsfw")
        if not ("ddim" in argdict["sched_name"] and argdict["eta"] > 0):
            argdict.pop("eta")
            argdict.pop("eta_seed")
        save_output(prompt, out, argdict, save_images, final_latent, display_with_cv2)
        if args.animate:
            # init frames by image list
            frames_by_image = []
            # one sublist per image
            for _ in range(len(PIL_animation_frames[0])):
                frames_by_image.append([])
            for images_of_step in PIL_animation_frames:
                for (image, image_sublist_index) in zip(images_of_step, range(len(images_of_step))):
                    frames_by_image[image_sublist_index].append(image)
            # remove the last frame of each sequence. It appears to be an extra, overly noised image
            frames_by_image = [list_of_frames[:-1] for list_of_frames in frames_by_image]
            save_animations(frames_by_image)
        return out,SUPPLEMENTARY

    if args.img_cycles > 0:
        correction_target = None
        # first frame is the initial image, if it exists.
        frames = [init_image] if init_image is not None else []
        try:
            for i in tqdm(range(args.img_cycles), position=0, desc="Image cycle"):
                if args.cycle_color_correction and correction_target is None and init_image is not None:
                    correction_target = image_to_correction_target(init_image)
                out, SUPPLEMENTARY = one_generation(args.prompt, init_image, display_with_cv2=IMAGE_CYCLE_DISPLAY_CV2, save_latents=args.image_cycle_save_individual, save_images=args.image_cycle_save_individual)
                if "cuda" in [IO_DEVICE, DIFFUSION_DEVICE]:
                    gc.collect()
                    torch.cuda.empty_cache()

                next_frame = out[0]
                frames.append(next_frame.copy())
                init_image = process_cycle_image(next_frame, args.img_rotation, args.img_center, args.img_translation, args.img_zoom, args.cycle_sharpen, args.width, args.height, args.cycle_color_correction, correction_target)

        except KeyboardInterrupt:
            # when an interrupt is received, cancel cycles and attempt to save any already generated images.
            pass
        if not args.image_cycle_save_individual:
            # if images are not saved on every step, save the final image separately
            try:
                argdict = SUPPLEMENTARY["io"]
                final_latent = SUPPLEMENTARY['latent']['final_latent']
                save_output(args.prompt, out, argdict, True, final_latent, False)
            except Exception as e:
                tqdm.write(f"Saving final image failed: {e}")
        save_animations([frames])
        exit()


    if args.prompts_file is not None:
        lines = [line.strip() for line in codecs.open("./artists.txt", "r", "utf-8-sig").readlines()]
        for line in tqdm(lines, position=0, desc="Input prompts"):
            one_generation(line, init_image)
    elif args.prompt is not None:
        one_generation(args.prompt, init_image)
    else:
        while True:
            prompt = input("Prompt> ").strip()
            try:
                one_generation(prompt, init_image)
            except KeyboardInterrupt:
                pass
            except Exception as e:
                print_exc()

def load_models(half_precision=False, unet_only=False):
    torch.no_grad() # We don't need gradients where we're going.
    tokenizer:transfomers_models.clip.tokenization_clip.CLIPTokenizer
    text_encoder:transfomers_models.clip.modeling_clip.CLIPTextModel
    unet:diffusers_models.unet_2d_condition.UNet2DConditionModel
    vae:diffusers_models.vae.AutoencoderKL

    unet_model_id = model_id
    vae_model_id = model_id
    use_auth_token_unet=True
    use_auth_token_vae=True

    unet_dir = os.path.join(models_local_dir, "unet")
    if os.path.exists(os.path.join(unet_dir, "config.json")) and os.path.exists(os.path.join(unet_dir, "diffusion_pytorch_model.bin")):
        use_auth_token_unet=False
        print("Using local unet model files!")
        unet_model_id = unet_dir
    vae_dir = os.path.join(models_local_dir, "vae")
    if os.path.exists(os.path.join(vae_dir, "config.json")) and os.path.exists(os.path.join(vae_dir, "diffusion_pytorch_model.bin")):
        use_auth_token_vae=False
        print("Using local vae model files!")
        vae_model_id = vae_dir

    # Load the UNet model for generating the latents.
    if half_precision:
        unet = UNet2DConditionModel.from_pretrained(unet_model_id, subfolder="unet", use_auth_token=use_auth_token_unet, torch_dtype=torch.float16, revision="fp16")
    else:
        unet = UNet2DConditionModel.from_pretrained(unet_model_id, subfolder="unet", use_auth_token=use_auth_token_unet)
    
    if unet_only:
        return unet

    # Load the autoencoder model which will be used to decode the latents into image space.
    if False:
        vae = AutoencoderKL.from_pretrained(vae_model_id, subfolder="vae", use_auth_token=use_auth_token_vae, torch_dtype=torch.float16)
    else:
        vae = AutoencoderKL.from_pretrained(vae_model_id, subfolder="vae", use_auth_token=use_auth_token_vae)
    # Load the tokenizer and text encoder to tokenize and encode the text. 
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    rate_nsfw = get_safety_checker()
    return tokenizer, text_encoder, unet, vae, rate_nsfw

def generate_segmented(tokenizer,text_encoder,unet,vae,IO_DEVICE="cpu",UNET_DEVICE="cuda",rate_nsfw=(lambda x: False),half_precision_latents=False):
    vae.to(IO_DEVICE)
    text_encoder.to(IO_DEVICE)
    unet.to(UNET_DEVICE)
    def generate_segmented_exec(prompt=["art"], width=512, height=512, steps=50, gs=7.5, seed=None, sched_name="pndm", eta=0.0, eta_seed=None, high_beta=False, animate=False, init_image=None, img_strength=0.5, save_latents=False):
        gc.collect()
        if "cuda" in [IO_DEVICE, DIFFUSION_DEVICE]:
            torch.cuda.empty_cache()
        with torch.no_grad():
            START_TIME = time()

            SUPPLEMENTARY = {
                "io" : {
                    "width" : 0,
                    "height" : 0,
                    "steps" : steps,
                    "gs" : gs,
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
            # 0xffff_ffff_ffff_ffff == 18446744073709551615
            seed = seed if seed is not None else random.randint(1, 18446744073709551615)
            eta_seed = eta_seed if eta_seed is not None else random.randint(1, 18446744073709551615)
            SUPPLEMENTARY["io"]["seed"] = seed
            SUPPLEMENTARY["io"]["eta_seed"] = eta_seed
            generator_eta = torch.manual_seed(eta_seed)
            generator_unet = torch.Generator(IO_DEVICE).manual_seed(seed)

            batch_size = len(prompt)
            sched_name = sched_name.lower().strip()
            SUPPLEMENTARY["io"]["sched_name"]=sched_name

            # text embeddings
            text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
            num_truncated_tokens = text_input.num_truncated_tokens
            SUPPLEMENTARY["io"]["text_readback"] = tokenizer.batch_decode(text_input.input_ids, cleanup_tokenization_spaces=False)
            # if a batch element had items truncated, the remaining token count is the negative of the amount of truncated tokens
            # otherwise, count the amount of <|endoftext|> in the readback. Sidenote: this will report incorrectly if the prompt is below the token limit and happens to contain "<|endoftext|>" for some unthinkable reason.
            SUPPLEMENTARY["io"]["remaining_token_count"] = [(- item) if item>0 else (SUPPLEMENTARY["io"]["text_readback"][idx].count("<|endoftext|>") - 1) for (item,idx) in zip(num_truncated_tokens,range(len(num_truncated_tokens)))]
            SUPPLEMENTARY["io"]["text_readback"] = [s.replace("<|endoftext|>","").replace("<|startoftext|>","") for s in SUPPLEMENTARY["io"]["text_readback"]]
            SUPPLEMENTARY["io"]["attention"] = text_input.attention_mask.cpu().numpy().tolist()
            # TODO: implement custom attention masks!

            text_embeddings = text_encoder(text_input.input_ids.to(IO_DEVICE))[0]
            max_length = text_input.input_ids.shape[-1]
            uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(IO_DEVICE))[0]
            #text_embeddings.to(UNET_DEVICE)
            #uncond_embeddings.to(UNET_DEVICE)
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            if half_precision_latents:
                    text_embeddings = text_embeddings.to(dtype=torch.float16)
            text_embeddings = text_embeddings.to(UNET_DEVICE)

            

            # schedulers
            if high_beta:
                # scheduler default
                beta_start = 0.0001
                beta_end = 0.02
            else:
                # stablediffusion default
                beta_start = 0.00085
                beta_end = 0.012
            if "lms" in sched_name:
                scheduler = LMSDiscreteScheduler(beta_start=beta_start, beta_end=beta_end, beta_schedule="scaled_linear", num_train_timesteps=1000)
            elif "pndm" in sched_name:
                # "for some models like stable diffusion the prk steps can/should be skipped to produce better results."
                scheduler = PNDMScheduler(beta_start=beta_start, beta_end=beta_end, beta_schedule="scaled_linear", num_train_timesteps=1000, skip_prk_steps=True) # <-- pipeline default
            elif "ddim" in sched_name:
                scheduler = DDIMScheduler(beta_start=beta_start, beta_end=beta_end, beta_schedule="scaled_linear", num_train_timesteps=1000)
            # set timesteps. Also offset, as pipeline does this
            accepts_offset = "offset" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if accepts_offset:
                scheduler_offset=1
                scheduler.set_timesteps(steps, offset=1)
            else:
                scheduler_offset=0
                scheduler.set_timesteps(steps)

            if isinstance(scheduler, LMSDiscreteScheduler):
                latents = latents * scheduler.sigmas[0]

            
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
                init_latents = vae.encode(init_image_tensor.to(IO_DEVICE)).sample()
                # apply inverse scaling
                init_latents = 0.18215 * init_latents
                init_latents = torch.cat([init_latents] * batch_size)
                init_timestep = int(steps * img_strength) + scheduler_offset
                init_timestep = min(init_timestep, steps)
                timesteps = scheduler.timesteps[-init_timestep]
                timesteps = torch.tensor([timesteps] * batch_size, dtype=torch.long, device=IO_DEVICE)
                noise = torch.randn(init_latents.shape, generator=generator_unet, device=IO_DEVICE)
                init_latents = scheduler.add_noise(init_latents, noise, timesteps)
                if half_precision_latents:
                    init_latents = init_latents.to(dtype=torch.float16)
                latents = init_latents.to(UNET_DEVICE)
                starting_timestep = max(steps - init_timestep + scheduler_offset, 0)

            # denoising loop!
            with autocast(UNET_DEVICE):
                #print(f"{unet.device} {latents.device} {text_embeddings.device}")
                for i, t in tqdm(enumerate(scheduler.timesteps[starting_timestep:]), position=1):
                    if animate:
                        SUPPLEMENTARY["latent"]["latent_sequence"].append(latents.clone().cpu())
                    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    latent_model_input = torch.cat([latents] * 2)
                    if isinstance(scheduler, LMSDiscreteScheduler):
                        sigma = scheduler.sigmas[i]
                        latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

                    # predict the noise residual
                    with torch.no_grad():
                        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)["sample"]

                    # perform guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + gs * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    if isinstance(scheduler, LMSDiscreteScheduler):
                        latents = scheduler.step(noise_pred, i, latents)["prev_sample"]
                    elif isinstance(scheduler, DDIMScheduler):
                        latents = scheduler.step(noise_pred, t, latents, eta=eta, generator=generator_eta)["prev_sample"]
                    else:
                        latents = scheduler.step(noise_pred, t, latents)["prev_sample"]

            # free up some now unused memory before attempting VAE decode!
            del noise_pred, noise_pred_text, noise_pred_uncond, latent_model_input, scheduler, text_embeddings, uncond_embeddings, uncond_input, text_input
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
            image = vae.decode(latents)

            # latents are decoded and copied (if requested) by now.
            del latents

            def to_pil(image, should_rate_nsfw:bool=True):
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
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
                    SUPPLEMENTARY["io"]["image_sequence"] = [to_pil(vae.decode(item.to(DIFFUSION_DEVICE) * (1 / 0.18215)), False)[0] for item in tqdm(SUPPLEMENTARY["latent"]["latent_sequence"], position=0, desc="Decoding animation latents")]
                torch.cuda.synchronize()
                vae.to(IO_DEVICE)
                unet.to(DIFFUSION_DEVICE)
            SUPPLEMENTARY["io"]["time"] = time() - START_TIME

            return pil_images, SUPPLEMENTARY
    return generate_segmented_exec

# can be called with perform_save=False to generate output image (grid_image when multiple inputs are given) and metadata
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

def get_safety_checker(device="cpu", safety_model_id = "CompVis/stable-diffusion-safety-checker"):
    safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)
    #safety_feature_extractor.to(device)
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
def interpolate_latents(latent_a, latent_b, steps=1):
    # TODO: Add overextension: get diff between a,b, then keep applying diff beyond the range between a,b (follow linear function outside of endpoints)
    a = latent_a[0]
    b = latent_b[0]
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
        initial_image.save(fp=image_basepath+f"{image_index}_a.webp", format='webp', append_images=frames[1:], save_all=True, duration=65, minimize_size=True, loop=1)
        initial_image.save(fp=image_basepath+f"{image_index}.gif", format='gif', append_images=frames[1:], save_all=True, duration=65)
        image_autogrid(frames).save(fp=image_basepath+f"{image_index}_grid.webp", format='webp', minimize_size=True)
    print(f"Saved files with prefix {image_basepath}")

def create_interpolation(a, b, steps, vae):
    al = load_latents_from_image(a)
    bl = load_latents_from_image(b)
    with torch.no_grad():
        interpolated = interpolate_latents(al,bl,steps)
    def to_pil(image):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images
    print("decoding...")
    with torch.no_grad():
        # processing images in target device one-by-one saves VRAM.
        image_sequence = to_pil(torch.stack([vae.decode(torch.stack([item.to(DIFFUSION_DEVICE)]) * (1 / 0.18215))[0] for item in tqdm(interpolated, position=0, desc="Decoding latents")]))
    save_animations([image_sequence])

if __name__ == "__main__":
    main()
