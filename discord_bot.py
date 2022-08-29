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

from generate import load_models, generate_segmented, save_output
from bot_token import get_token

model_id = "CompVis/stable-diffusion-v1-4"
UNET_DEVICE = "cuda"
IO_DEVICE = "cpu"
SAVE_OUTPUTS_TO_DISK = True
DEFAULT_HALF_PRECISION = True
COMMAND_PREFIX = "img>"

OUTPUTS_DIR = "outputs/generated"
INDIVIDUAL_OUTPUTS_DIR = os.path.join(OUTPUTS_DIR, "individual")
UNPUB_DIR = os.path.join(OUTPUTS_DIR, "unpub")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(INDIVIDUAL_OUTPUTS_DIR, exist_ok=True)
os.makedirs(UNPUB_DIR, exist_ok=True)

class command_task():
    def __init__(self,ctx:commands.context.Context,command:str) -> None:
        self.ctx=ctx
        self.command=command
        self.response:str="no response was returned"

class prompt_task():
    def __init__(self,ctx:commands.context.Context,prompt:str,w:int=512,h:int=512,steps:int=50,gs:float=7.5,seed:int=-1,eta:float=0.0,eta_seed:int=-1,init_img:Image=None,strength=0.75):
        self.ctx = ctx
        self.prompt = prompt
        self.w = w
        self.h = h
        self.steps = steps
        self.gs = gs
        self.seed = seed if seed >= 0 else random.randint(1, 18446744073709551615)
        self.eta_seed = eta_seed if eta_seed >= 0 else random.randint(1, 18446744073709551615)
        if eta > 0:
            self.eta = eta
            self.sched_name = "ddim"
        else:
            self.eta = 0.0
            self.sched_name = "pndm"
        self.datas = None
        self.response = None
        self.filename = None
        self.init_img = init_img
        self.strength = strength

    # params: (prompt=["art"], width=512, height=512, steps=50, gs=7.5, seed=None, sched_name="pndm", eta=0.0, eta_seed=None, high_beta=False, animate=False, init_image=None, img_strength=0.5, save_latents=False):
    def exec(self, generate_exec):
        prompts = [x.strip() for x in self.prompt.split("||")]
        out,SUPPLEMENTARY = generate_exec(
            prompt=prompts,
            width=self.w,
            height=self.h,
            steps=self.steps,
            gs=self.gs,
            seed=self.seed,
            sched_name=self.sched_name,
            eta=self.eta,
            eta_seed=self.eta_seed,
            init_image=self.init_img,
            img_strength=self.strength
        )
        argdict = SUPPLEMENTARY["io"]
        final_latent = SUPPLEMENTARY['latent']['final_latent']

        nsfw=argdict.get('nsfw', False)
        #img = out[0]
        self.filename = f"{'SPOILER_' if nsfw else 'img_'}generated"
        self.datas = []
        image_data = BytesIO()
        full_image, full_metadata, metadata_items = save_output(self.prompt, out, argdict, SAVE_OUTPUTS_TO_DISK, final_latent, False)
        full_image.save(image_data, "PNG", metadata=full_metadata)
        image_data.seek(0)
        self.datas.append(image_data)
        if len(out) > 1:
            individual_datas = []
            for _ in out:
                individual_datas.append(BytesIO())
            for img, metadata, bytes_item in zip(out,metadata_items,individual_datas):
                img.save(bytes_item, "PNG", metadata=metadata)
                bytes_item.seek(0)
            self.datas += individual_datas

        argdict.pop("image_sequence")
        argdict.pop("time")
        argdict.pop("attention")
        if argdict["width"] == 512:
            argdict.pop("width")
        if argdict["height"] == 512:
            argdict.pop("height")
        if not argdict["nsfw"]:
            argdict.pop("nsfw")
        if not ("ddim" in argdict["sched_name"] and argdict["eta"] > 0):
            argdict.pop("eta")
            argdict.pop("eta_seed")

        text_readback = argdict.pop("text_readback")
        if len(text_readback) == 1:
            text_readback = text_readback[0]
        self.response = f"{text_readback} ```{argdict}```"

task_queue = []
completed_tasks = []
currently_generating = False

def main():
    global task_queue
    global completed_tasks
    global currently_generating
    global precision_target_half
    #p = load_pipeline(model_id,device,True)
    precision_target_half = DEFAULT_HALF_PRECISION
    tokenizer, text_encoder, unet, vae, rate_nsfw = load_models(precision_target_half)
    generate_exec = generate_segmented(tokenizer=tokenizer,text_encoder=text_encoder,unet=unet,vae=vae,IO_DEVICE=IO_DEVICE,UNET_DEVICE=UNET_DEVICE,rate_nsfw=rate_nsfw)
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
                task.exec(generate_exec)
                completed_tasks.append(task)
                print(f"Done: '{task.prompt}', queued: {len(task_queue)}")
            elif isinstance(task,command_task):
                target = task.command
                """ # implementation appears to leak memory, requires further investigation.
                if target== "full":
                    if not precision_target_half:
                        task.response= "Already at full precision!"
                        pass # already at full precision
                    else:
                        precision_target_half = False
                        del unet
                        torch.cuda.empty_cache()
                        unet = load_models(precision_target_half, unet_only=True)
                        generate_exec = generate_segmented(tokenizer=tokenizer,text_encoder=text_encoder,unet=unet,vae=vae,IO_DEVICE="cpu",UNET_DEVICE="cuda",rate_nsfw=rate_nsfw)
                        task.response = "Switched to full precision!"
                elif target == "half":
                    if precision_target_half:
                        task.response = "Already at half precision!"
                        pass # already at half precision
                    else:
                        precision_target_half = True
                        del unet
                        torch.cuda.empty_cache()
                        unet = load_models(precision_target_half, unet_only=True)
                        generate_exec = generate_segmented(tokenizer=tokenizer,text_encoder=text_encoder,unet=unet,vae=vae,IO_DEVICE="cpu",UNET_DEVICE="cuda",rate_nsfw=rate_nsfw)
                        task.response = "Switched to half precision!"
                else:
                    task.response = f"Unknown request: {target}"
                """
                task.response = f"Ignoring: {target}. Currently disabled"
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
@bot.slash_command("generate")
async def generate(ctx, prompt:str):
    await generate_advanced(ctx, prompt)

@bot.slash_command("portrait")
async def generate_p(ctx, prompt:str):
    await generate_advanced(ctx, prompt, height=4)

@bot.slash_command("landscape")
async def generate_l(ctx, prompt:str):
    await generate_advanced(ctx, prompt, width=4)

@bot.slash_command("generate_advanced")
async def generate_advanced(ctx:commands.context.Context, prompt:str, width:int=0, height:int=0, seed:int=-1, gs:float=7.5, steps:int=50, eta:float=0.0, eta_seed:int=-1, strength:float=0.75):
    steps = steps if steps <= 150 else 150
    seed = seed if seed > 0 else -1
    eta_seed = eta_seed if eta_seed > 0 else -1
    global task_queue
    w = 512 + (width*64)
    h = 512 + (height*64)

    w = w if w > 64 else 64
    h = h if h > 64 else 64

    init_img = None
    if (len(ctx.message.attachments) > 0):
        try:
            url = ctx.message.attachments[0].url
            r = requests.get(url, stream=True)
            r.raw.decode_content = True # Content-Encoding
            init_img = Image.open(r.raw).convert("RGB")
        except Exception as e:
            init_img=None
            await ctx.reply(f"Unable to parse init image from message attachments: {e}. Running text-to-image only.", delete_after=10.0)

    task_queue.append(prompt_task(ctx, prompt=prompt.strip() ,w=w, h=h, steps=steps, gs=gs, seed=seed, eta=eta, eta_seed=eta_seed, init_img=init_img, strength=strength))
    await ctx.reply(f"Processing. Your prompt is number {len(task_queue)+ (1 if currently_generating else 0)} in queue.", delete_after=10.0)

@tasks.loop(seconds=1.0)
async def poll():
    global completed_tasks
    if len(completed_tasks) > 0:
        task:prompt_task
        task = completed_tasks.pop()
        if isinstance(task, prompt_task):
            files = [discord.File(fp=data, filename=f"{task.filename}_{i}.png") for (data, i) in zip(task.datas, range(len(task.datas)))]
            # await task.ctx.reply(task.response, file=discord.File(fp=task.data, filename=task.filename))
            await task.ctx.reply(task.response, files=files)
        elif isinstance(task, command_task):
            await task.ctx.reply(task.response, delete_after=10)
        else:
            print(f"WARNING: unknown task type found in completed_tasks queue: {type(task)}")
    else:
        #print(f"Idle: {len(completed_tasks)}")
        pass
poll.start()

"""
@tasks.loop(seconds=10.0)
async def set_status():
    global completed_tasks
    global bot
    await bot.change_presence(
        status = discord.Status.online,
        activity = discord.Activity(
            type=discord.ActivityType.listening, 
            name="prompt requests", 
            state="FP16" if precision_target_half else "FP32", 
            details=f"Queue: {len(completed_tasks) + (1 if currently_generating else 0)}"
        )
    )
set_status.start()
"""
bot.run(get_token())

"""
if __name__ == "__main__":
    main()
"""