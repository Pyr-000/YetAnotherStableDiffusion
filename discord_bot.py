import os
import requests
# before transformers/diffusers are imported
os.environ["TRANSFORMERS_VERBOSITY"] = "error" #if not verbose else "info"
os.environ["DIFFUSERS_VERBOSITY"] = "error" #if not verbose else "info"
from io import BytesIO
import random
import threading
from time import sleep
from typing import Tuple
import cv2
import numpy as np
from traceback import print_exc
from PIL import Image
import discord
from discord.ext import commands, tasks
from pathlib import Path

import generate
generate.PREPROCESS_DISPLAY_CV2 = False
generate.SAFETY_PROCESSOR_DISPLAY_MASK_CV2 = False
from generate import load_models, QuickGenerator, load_controlnet, image_autogrid, IMPLEMENTED_SCHEDULERS, IMPLEMENTED_GS_SCHEDULES, CONTROLNET_SHORTNAMES, IMPLEMENTED_CONTROLNET_PREPROCESSORS, input_to_seed
from tokens import get_discord_token

# set to True to enable the /reload command.
PERMIT_RELOAD = False
# permitted models act as a white-list. Dict keys specify the model name (used in /reload), while the value specifies the id/path respectively
# example: {"hub_v1.4":"CompVis/stable-diffusion-v1-4", "hub_v1.5":"runwayml/stable-diffusion-v1-5"}
permitted_model_ids = {"default_hub":generate.model_id}
# example: {"local_v1.4":"models/stable-diffusion-v1-4", "local_v1.5":"models/v1.5"}
permittel_local_model_paths = {"default_local":generate.models_local_dir}
# available controlnet shortnames
_controlnet_options_raw = list(CONTROLNET_SHORTNAMES.keys())
# available preprocessors: controlnet name -> relevant preprocessor name
_controlnet_options_preprocessors = {"canny":"canny","openpose":"detect_pose", "mlsd":"detect_mlsd", "hed":"detect_hed", "sd21-canny":"canny","sd21-openpose":"detect_pose", "sd21-hed":"detect_hed"}
CONTROLNET_PREPROCESS_PREFIX = "process_"
# split into two, because discord limits us to 25 options per parameter
controlnet_options_sd1 = [opt for opt in _controlnet_options_raw if not opt.startswith("sd21-")] + [f"{CONTROLNET_PREPROCESS_PREFIX}{x}" for x in _controlnet_options_preprocessors.keys() if not x.startswith("sd21-")]
controlnet_options_sd21 = [opt for opt in _controlnet_options_raw if opt.startswith("sd21-")] + [f"{CONTROLNET_PREPROCESS_PREFIX}{x}" for x in _controlnet_options_preprocessors.keys() if x.startswith("sd21-")]
def clamp_opt_length(option_list):
    return [x for x in option_list if x is not None][:24]
controlnet_options_sd1 = clamp_opt_length(controlnet_options_sd1)
controlnet_options_sd21 = clamp_opt_length(controlnet_options_sd21)
available_schedulers = clamp_opt_length(IMPLEMENTED_SCHEDULERS)
available_gs_schedulers = clamp_opt_length(IMPLEMENTED_GS_SCHEDULES)
# shortnames -> permitted loras (local path only)
PERMITTED_LORAS = {x.stem:str(x) for x in list(Path("./lora").glob("*.safetensors"))[:25]}

UNET_DEVICE = generate.DIFFUSION_DEVICE
IO_DEVICE = generate.IO_DEVICE
SAVE_OUTPUTS_TO_DISK = True
DEFAULT_HALF_PRECISION = True
COMMAND_PREFIX = "generate"
# --latents-half flag of generate.py: Memory usage reduction will be negligible (<1MB for 512x512), outputs will be slightly different.
USE_HALF_LATENTS = False
# if set to true, requests with an amount > 1 will always be generated sequentially to preserve VRAM. Will slow down generation speed for multiple images.
RUN_ALL_IMAGES_INDIVIDUAL = True
# if set to False, images are not checked for potential NSFW content. THIS SHOULD NOT BE USED unless additional content checking is employed downstream. Instead, set the appropriate processing level for your use-case.
FLAG_POTENTIAL_NSFW = True
# minimum safety processing level. See documentation of the '-spl' flag of generate.py
SAFETY_PROCESSING_LEVEL_NSFW_CHANNEL = 2
SAFETY_PROCESSING_LEVEL_SFW_CHANNEL = 5
# can be set to True to permit usage of the bot in SFW channels, with the selected safety processing level applied.
# ENABLE AT YOUR OWN RISK! This will require additional moderation and content screening. False negatives can occur in the safety checker, which can cause potentially NSFW content to be sent in SFW channels.
PERMIT_SFW_CHANNEL_USAGE = False
assert not PERMIT_SFW_CHANNEL_USAGE or (FLAG_POTENTIAL_NSFW and SAFETY_PROCESSING_LEVEL_SFW_CHANNEL >= 3)
# -as flag of generate.py: None to disable. Set UNET attention slicing slice size. 0 for recommended head_count//2, 1 for maximum memory savings
ATTENTION_SLICING = 1
# -co flag of generate.py: CPU offloading via accelerate. Should enable compatibility with minimal VRAM at the cost of speed.
CPU_OFFLOAD = False
# setting to skip last n CLIP (text encoder) layers. Recommended for some custom models.
CLIP_SKIP_LAYERS = 0
# limit images per message, can be set to 1 to circumvent discord tile-compacting multiple images from the same message. Values above 10 are ignored, as a limit of 10 images per message is set by discord.
IMAGES_PER_MESSAGE = 10
# input images grid: background settings ()
INPUT_IMAGES_BACKGROUND_COLOR = "green"
INPUT_IMAGES_SEPARATION = 4

def flatten_sublist(input_item):
    if not isinstance(input_item, list):
        return [input_item]
    else:
        concat = []
        for item in input_item:
            concat += flatten_sublist(item)
        return concat

class command_task():
    def __init__(self,ctx:discord.commands.context.ApplicationContext,type:str,command:dict) -> None:
        self.ctx=ctx
        self.type=type
        self.command=command
        self.response:str="no response was returned"

class prompt_task():
    def __init__(self,ctx:discord.commands.context.ApplicationContext,prompt:str, init_img:Image=None, generator_config:dict={}, controlnet:str=None, controlnet_input:Image=None,):
        self.ctx = ctx
        self.prompt = prompt
        self.init_img = init_img
        self.generator_config = generator_config
        self.generator_config["sequential_samples"] = RUN_ALL_IMAGES_INDIVIDUAL
        self.generator_config["attention_slicing"] = ATTENTION_SLICING
        self.generator_config["clip_skip_layers"] = CLIP_SKIP_LAYERS
        self.generator_config["display_with_cv2"] = False
        if controlnet is not None and controlnet.startswith(CONTROLNET_PREPROCESS_PREFIX):
            controlnet = controlnet[len(CONTROLNET_PREPROCESS_PREFIX):] # remove the prefix
            controlnet_preprocessor = _controlnet_options_preprocessors.get(controlnet,None) # acquire the relevant preprocessor name, if valid.
        else:
            controlnet_preprocessor = None
        self.controlnet_model_name = controlnet if controlnet in CONTROLNET_SHORTNAMES else None
        self.generator_config["controlnet_preprocessor"] = controlnet_preprocessor if controlnet_preprocessor in IMPLEMENTED_CONTROLNET_PREPROCESSORS else None
        self.controlnet_input = controlnet_input

        self.datas = None
        self.response = None
        self.filename = None

    def exec(self, generator:QuickGenerator):
        # reset configuration to defaults
        generator.init_config()
        # load any config args stored in the configuration dict
        # use default size (add an extra 256) of 768 for SD2 | Do not access through 'from generate import (...)', this would create a constant!
        if (self.generator_config["width"], self.generator_config["height"]) == (512,512):
            size_add = 0 if not generate.V_PREDICTION_MODEL else 256
            self.generator_config["width"] += size_add
            self.generator_config["height"] += size_add
        controlnet_model = load_controlnet(self.controlnet_model_name,UNET_DEVICE,CPU_OFFLOAD)
        generator.configure(**self.generator_config, controlnet=controlnet_model)
        # run generator
        out,SUPPLEMENTARY,save_return_data = generator.one_generation(
            prompt=self.prompt,
            init_image=self.init_img,
            save_images=SAVE_OUTPUTS_TO_DISK,
            controlnet_input=self.controlnet_input,
        )
        try: # clean up temporary controlnet model
            setattr(generator, "controlnet", None)
            del controlnet_model
        except Exception:
            pass
        # set response data on self
        self.create_response(out, SUPPLEMENTARY, save_return_data)

    def create_response(self, out, SUPPLEMENTARY, save_return_data):
        argdict = SUPPLEMENTARY["io"]
        nsfw=argdict.get('nsfw', False)
        self.filename = f"{'SPOILER_' if nsfw else 'img_'}generated"
        self.datas = []

        processed_controlnet_input = SUPPLEMENTARY["io"].pop("processed_controlnet_input")
        input_images = []
        if self.init_img is not None:
            input_images.append(self.init_img)
        if self.controlnet_model_name is not None and self.controlnet_input is not None:
            input_images.append(self.controlnet_input)
            if self.generator_config["controlnet_preprocessor"] is not None:
                # if the image was preprocessed, also append the processing result.
                input_images.append(processed_controlnet_input)
        input_images = [x.resize((self.generator_config["width"],self.generator_config["height"]), resample=Image.Resampling.LANCZOS) for x in input_images]
        input_image = None if len(input_images) < 1 else image_autogrid(input_images, fill_color=INPUT_IMAGES_BACKGROUND_COLOR, separation=INPUT_IMAGES_SEPARATION, frame_grid=True)
        if input_image is not None:
            init_image_data = BytesIO()
            input_image.save(init_image_data, "PNG")
            init_image_data.seek(0)
            self.datas.append(init_image_data)
        image_data = BytesIO()

        (full_image, full_metadata, metadata_items) = save_return_data
        full_image.save(image_data, "PNG", metadata=full_metadata)
        if (image_data.getbuffer().nbytes < (8388608 -512)) : # (keep 512 bytes of space. Discord apparently needs some extra bytes for itself within the 8MB limit.)
            image_data.seek(0)
            self.datas.append(image_data)
        else:
            print("INFO: Dropping grid image upload, as it exceeds the 8MB limit.")
        if len(out) > 1:
            individual_datas = []
            for _ in out:
                individual_datas.append(BytesIO())
            for img, metadata, bytes_item in zip(out,metadata_items,individual_datas):
                img.save(bytes_item, "PNG", metadata=metadata)
                bytes_item.seek(0)
            self.datas += individual_datas
        argdict["time"] = round(argdict["time"], 2)
        argdict.pop("attention", None)

        text_readback = argdict.pop("text_readback")
        self.response = f"{text_readback} ```{argdict}```"

task_queue = []
completed_tasks = []
currently_generating = False

def main():
    global task_queue
    global completed_tasks
    global currently_generating
    global precision_target_half
    global CPU_OFFLOAD
    global ATTENTION_SLICING
    global CLIP_SKIP_LAYERS
    global IMAGES_PER_MESSAGE
    IMAGES_PER_MESSAGE = max(min(IMAGES_PER_MESSAGE, 10),1)
    precision_target_half = DEFAULT_HALF_PRECISION
    generator = load_generator()
    print("loaded models!")
    while True:
        if len(task_queue) == 0:
            sleep(1)
            continue
        task = None
        try:
            currently_generating=True
            task = task_queue.pop(0)
            if isinstance(task, prompt_task):
                task.exec(generator)
                completed_tasks.append(task)
                print(f"Done: '{task.prompt}', queued: {len(task_queue)}")
            elif isinstance(task,command_task):
                if task.type == "reload" and PERMIT_RELOAD:
                    model_target = task.command["model"]
                    if model_target in permittel_local_model_paths:
                        pass
                        local_model_path = permittel_local_model_paths[model_target]
                        generate.models_local_dir = local_model_path
                    elif model_target in permitted_model_ids:
                        pass
                        local_model_id = permitted_model_ids[model_target]
                        generate.model_id = local_model_id
                        # disable local model override
                        generate.models_local_dir = None
                    else:
                        task.response = "Unable to find model despite passing model check!"
                        return
                    ATTENTION_SLICING = int(task.command["attention_slice"])
                    CPU_OFFLOAD = bool(task.command["offload"])
                    CLIP_SKIP_LAYERS = int(task.command["clip_skip_layers"])
                    del generator
                    generator = load_generator(lora_override=task.command.get("loras",None), lora_weight_override=task.command.get("lora_weights",None))
                    task.response = f"Re-loaded generator using model {model_target}. CPU offloading is {'enabled' if CPU_OFFLOAD else 'disabled'}"
                elif task.type == "default_negative":
                    generate.DEFAULT_NEGATIVE_PROMPT = task.command
                    task.response = f"Default prompt is now '{generate.DEFAULT_NEGATIVE_PROMPT}'" if generate.DEFAULT_NEGATIVE_PROMPT != "" else f"Default prompt has been reset."
                else:
                    task.response = f"Ignoring: {task.type} is not available."
                completed_tasks.append(task)
            else:
                print(f"WARNING: unknown task type found in task_queue: {type(task)}")
                completed_tasks.append(task)
            currently_generating=False
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print_exc()
            if task is not None:
                if isinstance(e, RuntimeError) and 'out of memory' in str(e).lower():
                    task.response = f"Out of memory for devices {UNET_DEVICE}(unet), {IO_DEVICE}(io)!"
                    task.response += ' CPU offloading currently not enabled. Consider loading the model with CPU_OFFLOAD=True' if not CPU_OFFLOAD else ''
                    task.response += ' Attention slicing is not set to the most memory efficient value. Consider loading the model with ATTENTION_SLICING=1' if ATTENTION_SLICING != 1 else ''
                else:
                    task.response = f"Something went horribly wrong: {e}"
                completed_tasks.append(task)

def load_generator(lora_override=None,lora_weight_override=None):
    cleanup_devices = [IO_DEVICE,UNET_DEVICE]
    if CPU_OFFLOAD:
        cleanup_devices.append(generate.OFFLOAD_EXEC_DEVICE)
    QuickGenerator.cleanup(cleanup_devices)
    tokenizer, text_encoder, unet, vae, rate_nsfw = load_models(precision_target_half, cpu_offloading=CPU_OFFLOAD, lora_override=lora_override, lora_weight_override=lora_weight_override)
    if not FLAG_POTENTIAL_NSFW:
        del rate_nsfw
        rate_nsfw = lambda x: False
    generator = QuickGenerator(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        unet=unet,
        vae=vae,
        IO_DEVICE=IO_DEVICE if not CPU_OFFLOAD else generate.OFFLOAD_EXEC_DEVICE,
        UNET_DEVICE=UNET_DEVICE if not CPU_OFFLOAD else generate.OFFLOAD_EXEC_DEVICE,
        rate_nsfw=rate_nsfw,
        use_half_latents=USE_HALF_LATENTS
    )
    return generator

# p = load_pipeline(model_id,device,True)
intents = discord.Intents.all()
bot = commands.Bot(COMMAND_PREFIX, case_insensitive=True)#, intents=intents)
th = threading.Thread(target=main)
th.daemon=True
th.start()

""" # disabled, as the current implementation appears to have a memory leak
@bot.command("full")
async def switch_f(ctx):
    task_queue.append(command_task(ctx,"full"))
    await ctx.reply(f"Attaching command to switch to full precision to queue! {len(task_queue)+ (1 if currently_generating else 0)} in queue.", delete_after=10.0)

@bot.command("half")
async def switch_h(ctx):
    task_queue.append(command_task(ctx,"half"))
    await ctx.reply(f"Attaching command to switch to half precision to queue! {len(task_queue)+ (1 if currently_generating else 0)} in queue.", delete_after=10.0)
"""

model_names = list(permittel_local_model_paths.keys()) + list(permitted_model_ids.keys())
@bot.slash_command(name="reload", description="Reload the generator with new parameters")
@discord.option("model",str,description="Model name (must be configured as permitted model name)",choices=[discord.OptionChoice(name=opt,value=opt) for opt in model_names],required=True)
@discord.option("offload",bool,description="Enable CPU offloading, slowing down generation but allowing significantly larger generations",required=False,default=False)
@discord.option("attention_slice",int,choices=[discord.OptionChoice(name=str(x), value=x) for x in [-1,0,1]],description="Set attention slicing: -1 -> None, 0 -> recommended, 1 -> max",required=False,default=1)
@discord.option("clip_skip_layers",int,description="Skip last n layers of CLIP text encoder. Recommended for some custom models",required=False,default=0)
@discord.option("lora_1",str,description="First LoRA to load",required=False,default=None,choices=[discord.OptionChoice(name=opt,value=opt) for opt in PERMITTED_LORAS.keys()])
@discord.option("lora_2",str,description="Second LoRA to load",required=False,default=None,choices=[discord.OptionChoice(name=opt,value=opt) for opt in PERMITTED_LORAS.keys()])
@discord.option("lora_3",str,description="Third LoRA to load",required=False,default=None,choices=[discord.OptionChoice(name=opt,value=opt) for opt in PERMITTED_LORAS.keys()])
@discord.option("lora_1_weight",float,description="Weight (alpha) of the first LoRA",required=False,default=0.5)
@discord.option("lora_2_weight",float,description="Weight (alpha) of the second LoRA",required=False,default=0.5)
@discord.option("lora_3_weight",float,description="Weight (alpha) of the third LoRA",required=False,default=0.5)
async def reload(ctx, model:str, offload:bool, attention_slice:int, clip_skip_layers:int, lora_1:str, lora_2:str, lora_3:str, lora_1_weight:float, lora_2_weight:float, lora_3_weight:float,):
    loras = []
    lora_weights = []
    for lora_name, lora_weight in [[lora_1,lora_1_weight],[lora_2,lora_2_weight],[lora_3,lora_3_weight]]:
        if lora_name is not None:
            loras.append(PERMITTED_LORAS[lora_name])
            lora_weights.append(lora_weight)
    if not (model in permitted_model_ids or model in permittel_local_model_paths):
        await ctx.send_response(f"Requested model is not available: {model}",delete_after=60)
    else:
        await ctx.send_response(f"Acknowledged. Your task is number {len(task_queue)+ (1 if currently_generating else 0)} in queue.")
        task_queue.append(command_task(ctx,type="reload",command={"model":model,"offload":offload,"attention_slice":attention_slice,"clip_skip_layers":clip_skip_layers,"loras":loras,"lora_weights":lora_weights}))

@bot.slash_command(name="default_negative", description="Set default negative prompt (used if none specified)")
@discord.option("negative_prompt",str,description="Default negative prompt, used if none is specified. Leave empty to reset.",required=False, default="")
async def default_negative(ctx, negative_prompt:str):
    await ctx.send_response(f"Acknowledged. Your task is number {len(task_queue)+ (1 if currently_generating else 0)} in queue.")
    task_queue.append(command_task(ctx,type="default_negative",command=negative_prompt))

@bot.slash_command(name="square", description="generate a default, square image (512x512)")
@discord.option("prompt",str,description="text prompt for generating. Multiple prompts can be specified in parallel, separated by two pipes ||",required=True)
@discord.option("init_image",discord.Attachment,description="Initial image for performing image-to-image",required=False,default=None)
async def square(ctx, prompt:str, init_image:discord.Attachment=None):
    reply = run_advanced(ctx, prompt, init_image=init_image)
    await ctx.send_response(reply)

@bot.slash_command(name="portrait", description="generate an image with portrait aspect ratio (512x768)")
@discord.option("prompt",str,description="text prompt for generating. Multiple prompts can be specified in parallel, separated by two pipes ||",required=True)
@discord.option("init_image",discord.Attachment,description="Initial image for performing image-to-image",required=False,default=None)
async def portrait(ctx, prompt:str, init_image:discord.Attachment=None):
    reply = run_advanced(ctx, prompt, height=4, init_image=init_image)
    await ctx.send_response(reply)

@bot.slash_command(name="landscape", description="generate an image with landscape aspect ratio (768x512)")
@discord.option("prompt",str,description="text prompt for generating. Multiple prompts can be specified in parallel, separated by two pipes ||",required=True)
@discord.option("init_image",discord.Attachment,description="Initial image for performing image-to-image",required=False,default=None)
async def landscape(ctx, prompt:str, init_image:discord.Attachment=None):
    reply = run_advanced(ctx, prompt, width=4, init_image=init_image)
    await ctx.send_response(reply)

@bot.slash_command(name="advanced", description="generate an image with custom parameters")
@discord.option("prompt",str,description="text prompt for generating. Multiple prompts can be specified in parallel, separated by two pipes ||",required=True)
@discord.option("width",int,description="Width in pixels or modifier offset in factor of 64 (0 -> 512 pixels, 2 -> 512+64*2, -2 -> 512-64*2)",required=False,default=0)
@discord.option("height",int,description="Height in pixels or modifier offset in factor of 64 (0 -> 512 pixels, 2 -> 512+64*2, -2 -> 512-64*2)",required=False,default=0)
@discord.option("seed",str,description="Initial noise seed for reproducing/modifying outputs (default: -1 will select a random seed)",required=False,default="-1")
@discord.option("gs",float,description="Guidance scale (increasing may increse adherence to prompt but decrease 'creativity'). Default: 9",required=False,default=9)
@discord.option("steps",int,description="Amount of sampling steps. More can help with detail, but increase computation time. Default: 50",required=False,default=50)
@discord.option("strength",float,description="Strength of img2img. 0.0 -> unchanged, 1.0 -> remade entirely. Requires valid image attachment.",required=False,default=0.75)
@discord.option("init_image",discord.Attachment,description="Initial image for performing image-to-image",required=False,default=None)
@discord.option("amount",int,description="Amount of images to batch at once.",required=False,default=1)
@discord.option("scheduler",str,description="Scheduler for the diffusion sampling loop.",choices=[discord.OptionChoice(name=opt,value=opt) for opt in available_schedulers],required=False,default="mdpms")
@discord.option("gs_schedule",str,description="Variable guidance scale schedule. Default -> constant scale.",choices=[discord.OptionChoice(name=opt,value=opt) for opt in available_gs_schedulers if opt is not None]+[discord.OptionChoice(name="None",value="None")],required=False,default=None)
@discord.option("guidance_rescale",float,description="Strength of prediction distribution correction during sampling (between 0 and 1).",required=False,default=0.66)
@discord.option("static_length",int,description="Static embedding length. Disables dynamic length mode. 77 to reproduce previous behavior",required=False,default=-1)
#@discord.option("mix_concatenate",bool,description="When mixing prompts, concatenate the embeddings instead of computing their weighted sum.",required=False,default=False)
#@discord.option("eta",float,description="Higher 'eta' -> more random noise during sampling. Ignored unless scheduler=ddim",required=False,default=0.0)
#@discord.option("eta_seed",str,description="Acts like 'seed', but only applies to the sampling noise for eta > 0.",required=False,default="-1")
@discord.option("controlnet",str,description="Controlnet model for closer control over the generated image",required=False,default=None,choices=[discord.OptionChoice(name=opt,value=opt) for opt in controlnet_options_sd1 if opt is not None]+[discord.OptionChoice(name="None",value="None")])
@discord.option("controlnet_sd2",str,description="Controlnet model options for SD2.1",required=False,default=None,choices=[discord.OptionChoice(name=opt,value=opt) for opt in controlnet_options_sd21 if opt is not None]+[discord.OptionChoice(name="None",value="None")])
@discord.option("controlnet_input",discord.Attachment,description="Input image for the controlnet",required=False,default=None)
@discord.option("controlnet_strength",float,description="Strength (scale) of controlnet guidance",required=False,default=1.0)
@discord.option("controlnet_schedule",str,description="Variable guidance scale schedule for controlnets. Default -> constant scale.",choices=[discord.OptionChoice(name=opt,value=opt) for opt in available_gs_schedulers if opt is not None]+[discord.OptionChoice(name="None",value="None")],required=False,default=None)
@discord.option("second_pass_resize",float,description="Resize factor for performing two-pass generation. Enabled when >1.",required=False,default=1)
@discord.option("second_pass_steps",int,description="Amount of second pass sampling steps when two-pass generation is selected.",required=False,default=50)
@discord.option("second_pass_ctrl",bool,description="Use a specified controlnet instead of img2img for the second pass.",required=False,default=False)
@discord.option("use_karras_sigmas",bool,description="Use the Karras sigma schedule",required=False,default=True)
@discord.option("lora_schedule",str,description="Variable strength schedule for loaded LoRA-like models. Default -> constant scale.",choices=[discord.OptionChoice(name=opt,value=opt) for opt in available_gs_schedulers if opt is not None]+[discord.OptionChoice(name="None",value="None")],required=False,default=None)

async def advanced(
    ctx:discord.commands.context.ApplicationContext, prompt:str, width:int=0, height:int=0, seed:str="-1", gs:float=9, steps:int=50, strength:float=0.75, init_image:discord.Attachment=None, amount:int=1,
    scheduler:str="mdpms", gs_schedule:str=None, guidance_rescale:float=0.66, static_length:int=-1, # mix_concatenate:bool=False, eta:float=0.0, eta_seed:str="-1",
    controlnet:str=None, controlnet_sd2:str=None, controlnet_input:discord.Attachment=None, controlnet_strength:float=1, controlnet_schedule:str=None,
    second_pass_resize:float=1, second_pass_steps:int=50, second_pass_ctrl:bool=False, use_karras_sigmas:bool=True, lora_schedule:str=None,
):
    reply = run_advanced(**locals())
    await ctx.send_response(reply)

def run_advanced(
    ctx:discord.commands.context.ApplicationContext, prompt:str, width:int=0, height:int=0, seed:str="-1", gs:float=9, steps:int=50, strength:float=0.75, init_image:discord.Attachment=None, amount:int=1,
    scheduler:str="mdpms", gs_schedule:str=None, static_length:int=-1, mix_concatenate:bool=False, eta:float=0.0, eta_seed:str="-1",
    controlnet:str=None, controlnet_sd2:str=None, controlnet_input:discord.Attachment=None, controlnet_strength:float=1, controlnet_schedule:str=None,
    guidance_rescale:float=0.66, second_pass_resize:float=1, second_pass_steps:int=50, second_pass_ctrl:bool=False, use_karras_sigmas:bool=True, lora_schedule:str=None,
):
    if hasattr(ctx.channel, "is_nsfw") and not ctx.channel.is_nsfw():
        if not PERMIT_SFW_CHANNEL_USAGE:
            return "Refusing, as channel is not marked as NSFW and usage in SFW channels is disabled!"
        else:
            safety_processing_level = SAFETY_PROCESSING_LEVEL_SFW_CHANNEL
    else:
        safety_processing_level = SAFETY_PROCESSING_LEVEL_NSFW_CHANNEL
    global task_queue

    # process value limits
    steps = steps if steps > 0 else 1
    steps = steps if steps <= 150 else 150
    amount = amount if amount <= 16 else 16
    seed = input_to_seed(seed)
    eta_seed = input_to_seed(eta_seed)
    # if values above 64 are given, presume them to be pixel values
    w = (512 + (width*64)) if width < 64 else width
    h = (512 + (height*64)) if height < 64 else height
    w = w if w > 64 else 64
    h = h if h > 64 else 64
    scheduler = None if scheduler == "None" else scheduler
    static_length = None if static_length < 3 else static_length

    init_img, additional_text = get_init_image_from_attachment(init_image)
    controlnet_input, more_add_text = get_init_image_from_attachment(controlnet_input)
    additional_text = ("" if additional_text is None else additional_text) + ("" if more_add_text is None else more_add_text)

    # keyword args of QuickGenerator.configure(). Unspecified args will remain as the default.
    generator_config = {
        "width":w,
        "height":h,
        "steps":steps,
        "guidance_scale":gs,
        "sample_count":amount,
        "seed":seed,
        "sched_name":scheduler,
        "ddim_eta":eta,
        "eta_seed":eta_seed,
        "strength":strength,
        "gs_scheduler":gs_schedule,
        "static_length":static_length,
        "mix_mode_concat":mix_concatenate,
        "controlnet_strength":controlnet_strength,
        "controlnet_strength_scheduler":controlnet_schedule,
        "safety_processing_level":safety_processing_level,
        "guidance_rescale":guidance_rescale,
        "second_pass_resize":second_pass_resize,
        "second_pass_steps":second_pass_steps,
        "second_pass_use_controlnet":second_pass_ctrl,
        "use_karras_sigmas":use_karras_sigmas,
        "lora_schedule":lora_schedule,
    }

    controlnet = controlnet if controlnet is not None else controlnet_sd2
    task_queue.append(prompt_task(ctx, prompt=prompt, init_img=init_img, generator_config=generator_config, controlnet=controlnet, controlnet_input=controlnet_input))
    return f"Processing. Your prompt is number {len(task_queue)+ (1 if currently_generating else 0)} in queue. {additional_text}".strip()

def get_init_image_from_attachment(attachment):
    init_img = None
    additional = ""
    if attachment is not None:
        try:
            if not attachment.content_type.startswith("image"):
                additional = f"Attachment must be an image! Got type {attachment.content_type}."
            elif attachment.size > 8388608:
                additional = "Attached file is too large! Try again with a smaller file (limit:8MB)."
            else:
                url = attachment.url
                r = requests.get(url, stream=True)
                r.raw.decode_content = True # Content-Encoding
                init_img = Image.open(r.raw).convert("RGB")
        except Exception as e:
            init_img=None
            additional =  f"Unable to parse init image from message attachments: {e}. Running text-to-image only."
    return init_img,additional

@tasks.loop(seconds=1.0)
async def poll():
    global completed_tasks
    if len(completed_tasks) > 0:
        task:prompt_task
        task = completed_tasks.pop()
        tag = task.ctx.author.mention
        if isinstance(task, prompt_task):
            if task.datas is not None:
                datas = [data for data in task.datas if (data.getbuffer().nbytes < (8388608 -512))]
                files = [discord.File(fp=data, filename=f"{task.filename}_{i}.png") for (data, i) in zip(datas, range(len(datas)))]
                if len(datas) < len(files):
                    print(f"Dropped: {len(datas)-len(files)} due to exceeding the 8MB limit")
                i=0
                chunk = []
                chunk_size = 0
                while len(files)>0:
                    next_file = files.pop(0)
                    next_size = next_file.fp.getbuffer().nbytes
                    # file size limit seems to be applied to the sum of all files, instead of to each file individually. Keeping to <=10 files within the limit (individually) can still cause HTTP413
                    if (next_size + chunk_size > (8388608 -512)) or (len(chunk) >= IMAGES_PER_MESSAGE):
                        # if too much data would be present, send accumulated data and start a new chunk.
                        await (task.ctx.edit if i==0 else task.ctx.send_followup)(content=f"{task.response}"[:1999] if i==0 else f"[{i}]", files=chunk)
                        sleep(0.2)
                        chunk = [next_file]
                        chunk_size = next_size
                        i += 1
                    else:
                        chunk.append(next_file)
                        chunk_size += next_size
                if len(chunk) > 0:
                    await (task.ctx.edit if i==0 else task.ctx.send_followup)(content=f"{task.response}"[:1999] if i==0 else f"[{i}]", files=chunk)
            else:
                await task.ctx.edit(content=f"{task.response}"[:1999], files=None, delete_after=60)
        elif isinstance(task, command_task):
            await task.ctx.edit(content=f"{task.response}"[:1999])
        else:
            print(f"WARNING: unknown task type found in completed_tasks queue: {type(task)}")
        del task
    else:
        #print(f"Idle: {len(completed_tasks)}")
        pass
poll.start()

"""
@tasks.loop(seconds=10.0)
async def set_status():
    global completed_tasks
    # await bot.change_presence(status = discord.Status.online,activity=discord.Game(f"with diffusers. Queue: {len(completed_tasks) + (1 if currently_generating else 0)}"))
set_status.start()
"""
bot.run(get_discord_token())

"""
if __name__ == "__main__":
    main()
"""
