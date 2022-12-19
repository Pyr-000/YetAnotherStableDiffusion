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

import generate
from generate import load_models, QuickGenerator, IMPLEMENTED_SCHEDULERS, IMPLEMENTED_GS_SCHEDULES
from tokens import get_discord_token

# set to True to enable the /reload command.
PERMIT_RELOAD = False
# permitted models act as a white-list. Dict keys specify the model name (used in /reload), while the value specifies the id/path respectively
# example: {"hub_v1.4":"CompVis/stable-diffusion-v1-4", "hub_v1.5":"runwayml/stable-diffusion-v1-5"}
permitted_model_ids = {"default_hub":generate.model_id}
# example: {"local_v1.4":"models/stable-diffusion-v1-4", "local_v1.5":"models/v1.5"}
permittel_local_model_paths = {"default_local":generate.models_local_dir}

UNET_DEVICE = "cuda"
IO_DEVICE = "cuda"
SAVE_OUTPUTS_TO_DISK = True
DEFAULT_HALF_PRECISION = True
COMMAND_PREFIX = "generate"
# --latents-half flag of generate.py: Memory usage reduction will be negligible (<1MB for 512x512), outputs will be slightly different.
USE_HALF_LATENTS = False
# if set to true, requests with an amount > 1 will always be generated sequentially to preserve VRAM. Will slow down generation speed for multiple images.
RUN_ALL_IMAGES_INDIVIDUAL = False
# if set to False, images are not checked for potential NSFW content. This disables flagging them as spoilers and speeds up generation slightly.
FLAG_POTENTIAL_NSFW = True
# -as flag of generate.py: None to disable. Set UNET attention slicing slice size. 0 for recommended head_count//2, 1 for maximum memory savings
ATTENTION_SLICING = 1
# -co flag of generate.py: CPU offloading via accelerate. Should enable compatibility with minimal VRAM at the cost of speed.
CPU_OFFLOAD = False

# create a string from a list, while removing duplicate string items in a row.
def collapse_representation(item_list):
    collapsed = []
    for item in item_list:
        if (not len(collapsed) > 0) or collapsed[-1] != str(item):
            collapsed.append(str(item))
    if len(collapsed) == 1:
        return collapsed[0]
    else:
        # this also catches len(collapsed)==0, returning "[]"
        return str(collapsed)


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
    def __init__(
        self,
        ctx:discord.commands.context.ApplicationContext,prompt:str, init_img:Image=None, generator_config:dict={}):
        self.ctx = ctx
        self.prompt = prompt
        self.init_img = init_img
        self.generator_config = generator_config
        self.generator_config["sequential_samples"] = RUN_ALL_IMAGES_INDIVIDUAL
        self.generator_config["attention_slicing"] = ATTENTION_SLICING
        self.generator_config["display_with_cv2"] = False

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
        generator.configure(**self.generator_config)
        # run generator
        out,SUPPLEMENTARY,save_return_data = generator.one_generation(
            prompt=self.prompt,
            init_image=self.init_img,
            save_images=SAVE_OUTPUTS_TO_DISK
        )
        # set response data on self
        self.create_response(out, SUPPLEMENTARY, save_return_data)

    def create_response(self, out, SUPPLEMENTARY, save_return_data):
        argdict = SUPPLEMENTARY["io"]
        nsfw=argdict.get('nsfw', False)
        self.filename = f"{'SPOILER_' if nsfw else 'img_'}generated"
        self.datas = []

        if self.init_img is not None:
            init_image_data = BytesIO()
            self.init_img.save(init_image_data, "PNG")
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

        text_readback = collapse_representation(argdict.pop("text_readback"))
        argdict["remaining_token_count"] = collapse_representation(argdict["remaining_token_count"])
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
                    del generator
                    generator = load_generator()
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
                task.response = f"Something went horribly wrong: {e}"
                completed_tasks.append(task)

def load_generator():
    cleanup_devices = [IO_DEVICE,UNET_DEVICE]
    if CPU_OFFLOAD:
        cleanup_devices.append(generate.OFFLOAD_EXEC_DEVICE)
    QuickGenerator.cleanup(cleanup_devices)
    tokenizer, text_encoder, unet, vae, rate_nsfw = load_models(precision_target_half, cpu_offloading=CPU_OFFLOAD)
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
async def square(ctx, model:str, offload:bool, attention_slice:int):
    if not (model in permitted_model_ids or model in permittel_local_model_paths):
        await ctx.send_response(f"Requested model is not available: {model}",delete_after=60)
    else:
        await ctx.send_response(f"Acknowledged. Your task is number {len(task_queue)+ (1 if currently_generating else 0)} in queue.")
        task_queue.append(command_task(ctx,type="reload",command={"model":model,"offload":offload,"attention_slice":attention_slice}))

@bot.slash_command(name="default_negative", description="Set default negative prompt (used if none specified)")
@discord.option("negative_prompt",str,description="Default negative prompt, used if none is specified. Leave empty to reset.",required=False, default="")
async def square(ctx, negative_prompt:str):
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
@discord.option("scheduler",str,description="Scheduler for the diffusion sampling loop.",choices=[discord.OptionChoice(name=opt,value=opt) for opt in IMPLEMENTED_SCHEDULERS],required=False,default="mdpms")
@discord.option("gs_schedule",str,description="Variable guidance scale schedule. Default -> constant scale.",choices=[discord.OptionChoice(name=opt,value=opt) for opt in IMPLEMENTED_GS_SCHEDULES if opt is not None]+[discord.OptionChoice(name="None",value="None")],required=False,default=None)
@discord.option("eta",float,description="Higher 'eta' -> more random noise during sampling. Ignored unless scheduler=ddim",required=False,default=0.0)
@discord.option("eta_seed",str,description="Acts like 'seed', but only applies to the sampling noise for eta > 0.",required=False,default="-1")
async def advanced(ctx:discord.commands.context.ApplicationContext, prompt:str, width:int=0, height:int=0, seed:str="-1", gs:float=9, steps:int=50, strength:float=0.75, init_image:discord.Attachment=None, amount:int=1, scheduler:str="mdpms", gs_schedule:str=None, eta:float=0.0, eta_seed:str="-1"):
    reply = run_advanced(**locals())
    await ctx.send_response(reply)

def run_advanced(ctx:discord.commands.context.ApplicationContext, prompt:str, width:int=0, height:int=0, seed:str="-1", gs:float=9, steps:int=50, strength:float=0.75, init_image:discord.Attachment=None, amount:int=1, scheduler:str="mdpms", gs_schedule:str=None, eta:float=0.0, eta_seed:str="-1"):
    if hasattr(ctx.channel, "is_nsfw") and not ctx.channel.is_nsfw():
        return "Refusing, as channel is not marked as NSFW. While images are sent as spoilers if potential NSFW content is detected, there is no NSFW filter in effect."
    global task_queue

    # process value limits
    steps = steps if steps > 0 else 1
    steps = steps if steps <= 150 else 150
    amount = amount if amount <= 16 else 16
    try:
        seed = int(seed)
    except:
        seed = -1
    try:
        eta_seed = int(eta_seed)
    except:
        eta_seed = -1
    seed = seed if seed > 0 else None
    eta_seed = eta_seed if eta_seed > 0 else None
    # if values above 64 are given, presume them to be pixel values
    w = (512 + (width*64)) if width < 64 else width
    h = (512 + (height*64)) if height < 64 else height
    w = w if w > 64 else 64
    h = h if h > 64 else 64
    scheduler = None if scheduler == "None" else scheduler

    init_img, additional_text = get_init_image_from_attachment(init_image)

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
    }

    task_queue.append(prompt_task(ctx, prompt=prompt, init_img=init_img, generator_config=generator_config))
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
                    if (next_size + chunk_size > (8388608 -512)) or (len(chunk) >= 10):
                        # if too much data would be present, send accumulated data and start a new chunk.
                        await (task.ctx.edit if i==0 else task.ctx.send_followup)(content=f"{task.response}" if i==0 else f"[{i}]", files=chunk)
                        sleep(0.2)
                        chunk = [next_file]
                        chunk_size = next_size
                        i += 1
                    else:
                        chunk.append(next_file)
                        chunk_size += next_size
                if len(chunk) > 0:
                    await (task.ctx.edit if i==0 else task.ctx.send_followup)(content=f"{task.response}" if i==0 else f"[{i}]", files=chunk)
            else:
                await task.ctx.edit(content=f"{task.response}", files=None, delete_after=60)
        elif isinstance(task, command_task):
            await task.ctx.edit(content=f"{task.response}")
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
