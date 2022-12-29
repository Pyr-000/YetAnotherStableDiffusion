# Yet Another StableDiffusion Implementation
Stable Diffusion script(s) based on huggingface diffusers. Comes with extra configurability and some bonus features, a single script for accessing every functionality, and example code for a discord bot demonstrating how it can be imported in other scripts.
## Recent changes to requirements
- Updating `diffusers`, `transformers`, `huggingface-hub` and `accelerate` is recommended.
- `diffusers` should be updated. `accelerate` can now be used to perform automatic CPU offloading.
- Now requires `scikit-image` package to perform color correction during image cycling

# Python Installation
## Install `pytorch` (skip when using a pre-existing StableDiffusion-compatible environment)
Get the correct installation for your hardware at https://pytorch.org/get-started/locally/.

When installing pytorch with CUDA support, the `conda` install command will install cudatoolkit as well.

When installing with `pip` directly (e.g. for non-conda environments), CUDA toolkit may also need to be manually installed. CUDA toolkit can be found at: https://developer.nvidia.com/cuda-downloads.
#
## Install additional dependencies
```shell
pip install --upgrade diffusers transformers scipy ftfy opencv-python huggingface_hub scikit-image accelerate
```
Most pre-existing StableDiffusion-compatible environments will already have some of these installed.
#
# Model Installation
## Local installation via git
Git must be installed on your system. Equivalent to [Option B: Manual model install](#option-b-manual-model-install), but automated.
Note the license terms ([CreativeML OpenRAIL++-M](https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/LICENSE-MODEL)).

From the root directory of this repository, clone your model repository into `models/` (this example will clone StableDiffusion v2.1):
```shell
git lfs clone https://huggingface.co/stabilityai/stable-diffusion-2-1 models/stable-diffusion-2-1 --progress -X "v2-1_768-ema-pruned.ckpt","v2-1_768-nonema-pruned.ckpt"
```
This may take a while.
Using 'git lfs clone' will yield a deprecation warning, however, cloning through 'git clone' may fail to display any progress information on some platforms.

#
## Alternative model installation options
Models can either be automatically installed by providing a huggingface token, or manually installed by downloading them from huggingface yourself.

### A Note on StableDiffusion version 2.x+:
Currently, only [Option B: manual model install](#option-b-manual-model-install) is supported for SD2.x models. StableDiffusion v2.1 can be acquired from https://huggingface.co/stabilityai/stable-diffusion-2-1.
In addition to the `unet` and `vae` folders, the `scheduler`, `text_encoder` and `tokenizer` folders must also be added to the model directory, together with their respective files:
- `models/*/scheduler/`: `scheduler_config.json`
- `models/*/text_encoder/`: `config.json` and `pytorch_model.bin`
- `models/*/tokenizer/`: `merges.txt`, `special_tokens_map.json`, `tokenizer_config.json` and `vocab.json`

If `scheduler/scheduler_config.json` is not provided, the model will be presumed to not be a v_prediction model (this will cause issues with anything but the _base_ variant of SD2.x).
If `text_encoder` and `tokenizer` do not provide the required files, the model will be loaded with `openai/clip-vit-large-patch14`, which is used in SD1.x-style models. This is incompatible with SD2.x.

### Option A: Automatic model install via huggingface:
You should have read and accepted the license terms ([CreativeML OpenRAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license)) of the relevant StableDiffusion model repository at https://huggingface.co/CompVis/stable-diffusion-v1-4. Manually logging in and accepting the license on the huggingface webpage is no longer required.

Model installation should occur automatically, even without being logged into huggingface hub.

<!--If you have already logged into huggingface-cli on your machine, you can skip the model installation step. Models will be downloaded automatically.

Otherwise, get a valid token with read access for your huggingface account, accessible at https://huggingface.co/settings/tokens. You can then:-->
As manually accepting the license on the huggingface webpage should no longer be required, you should no longer need to log in. If your online model repository requires access credentials, you can provide them via the following:
- Either paste the token into the `tokens.py` file: `HUGGINGFACE_TOKEN = "your token here"`
- Or log into huggingface-cli with your token (run `huggingface-cli login` in a terminal while your python environment is activated). This login should remain stored in your user directory until you log out with the cli, independent of your python environment.

### Option B: Manual model install:
- Navigate to https://huggingface.co/CompVis/stable-diffusion-v1-4/. Note the license terms ([CreativeML OpenRAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license)) (logging in to accept the license should no longer be required).
- Head to the `Files and versions` tab, and navigate to `stable-diffusion-v1-4/unet`
  - Download both `config.json` and `diffusion_pytorch_model.bin`
  - place both files in `models/stable-diffusion-v1-4/unet`
- Repeat for the VAE model: In the `Files and versions` tab, navigate to `stable-diffusion-v1-4/vae`
  - Download both `config.json` and `diffusion_pytorch_model.bin`
  - Place the files in `models/stable-diffusion-v1-4/vae`
- It is recommended to keep the branch on the default: `main` instead of switching to `fp16`. The fp32 weights are larger (3.44GB instead of 1.72GB), but can be loaded for both full precision (fp32) and half precision (fp16) use. 
- Note: The checkpoint files for diffusers are not the same as standard StableDiffusion checkpoint files (e.g. sd-v1-4.ckpt). They can not be copied over directly.
  - If required, custom (monolithic) model checkpoints designated for the "standard StableDiffusion" implementation can be converted to separate models for use with diffusers using their provided conversion script: [diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py](https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py), available from the [huggingface diffusers repository](https://github.com/huggingface/diffusers).
- Note: The VAE files can instead be replaced by an StabilityAIs improved `ft-EMA` / `ft-MSE` VAEs, which are available under [huggingface.co/stabilityai/sd-vae-ft-mse/tree/main](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main)

#
# Usage & Features
- Features include text-to-image, image-to-image (including cycling), and options for precision/devices to control generation speed and memory usage.
  - Additional memory optimisation options include (automatic) sequential batching, attention slicing and CPU offloading (see: [Optimisation settings](#device-performace-and-optimization-settings))
- Animating prompt interpolations is possible for both image-to-image cycling and text-to-image (`-cfi`, see: [Additional Flags](#additional-flags)).
- `prompts`, `seeds`, and other relevant arguments will be stored in PNG metadata by default.
- For every generated output, an image will be saved in `outputs/generated`. When multiple images are batched at once, this singular image will be a grid, with the individual images being saved in `outputs/generated/individual`. Additionally, each generated output will produce a text file in `outputs/generated/unpub` of the same filename, containing the prompt and additional arguments. This can be used for maintaining an external image gallery.
  - Optionally, a different output directory can be specified (`-od`, see: [Additional flags](#additional-flags)).
- The NSFW check is enabled by default, but will only print a warning message and attach a metadata label, leaving the output images untouched. It can also be fully disabled via a commandline flag (see: [Additional Flags](#additional-flags))
- Cycling through iterations of image-to-image comes with a mitigation for keeping the color scheme unchanged, preventing effects such as 'magenta shifting'. (`-icc`, see: [Image to image cycling](#image-to-image-cycling))
- "Textual Inversion Concepts", custom prompt embeddings, from https://huggingface.co/sd-concepts-library can be placed in `models/concepts` (only the .bin file is required, other files are ignored). They will be loaded into the text encoder automatically. `.pt`-style custom embeddings are also supported in the same way.
- Prompts can be mixed using prompt weights: When prompts are separated by `;;`, their representation within the text encoder space will be averaged. Custom prompt weights can be set by putting a number between the `;;` trailing the prompt. If no value is specified, the default value of 1 will be used as the prompt weight. 
  - Example: `"Painting of a cat ;3; Photograph of a cat ;1;"` Will yield the text representation of `3/4 * "Painting of a cat" + 1/4 "Photograph of a cat"`
  - By default, all negative subprompts will be mixed accoring to their weight, and used in place of the unconditional embedding for classifier free guidance (standard "negative prompts"). In this case, the difference in weights between the positive and negative prompts is not considered, as this is given by the guidance scale (`-cs`).
  - Alternatively, prompts with negative weight values can be directly mixed into the prompt itself, leaving the unconditional embedding untouched (`-mn`, [Additional Flags](#additional-flags)). In this case, negative prompts are not directly 'subtracted'. Instead, the prompt is amplified in its difference from the negative prompts (moving away from the prompt, in the opposite direction of the negative prompt). Relative weight values between positive and negative prompts are considered. This way of applying negative prompts tends to be far more chaotic, but can yield interesting results. In this mode, a loose list of unwanted attributes as a negative prompt will usually perform worse than a description of the desired image together with negative attributes.
- Prompts can be specified with an (individual) CLIP-skip setting by appending a trailing `{cls<n>}` for a setting of `n`. This will skip the last *n* layers of CLIP, the text encoder. Increasing this value will reduce the "amount of processing"/"depth of interpretation" performed by the text encoder. A value of `0` is equivalent to specifying no CLIP-skip setting.
  - Example: `"Painting of a cat{cls2}"` will encode "Painting of a cat", while skipping the final two layers of the text encoder.
  - When combined with prompt mixing or negative prompts (`;;`, see above), the prompt separator must be specified after the CLIP-skip setting. The skip setting is independent for each sub-prompt.
    - Example: `"Painting of a cat{cls1};3; Photograph of a cat{cls2};1;"`
    - As shown in the example, this can also be used to mix text prompts with themselves under different skip settings, or for interpolating between the same prompt under different skip settings (see: [Cycling](#image-to-image-cycling) and `-cfi` under [Additional Flags](#additional-flags))
  - For some custom models, using a specific CLIP-skip setting by default is recommended. The default value used when none is specified can be set via `-cls` (see: [Additional Flags](#additional-flags)).

# 
## text to image
By default, `generate.py` will run in text-to-image mode. If launched without a prompt, it will enter a prompt input loop until the program is manually terminated. Alternatively a prompt can be specified as a commandline argument:
```shell
python generate.py "a painting of a painter painting a painting"
```
### Multiple prompts
- Multiple prompts can be specified this way, separated by two pipes (`||`). Any prompts specified this way will be multiplied by the set number of samples (`-n` or `--n-samples`). As all prompts and samples are run as a single batch, in parallel, increasing their amount will increase the memory requirement. If allocating memory for processing the outputs in parallel fails, the generator will attempt to generate the images sequentially instead. (see: [`-seq`](#device-performace-and-optimization-settings))
- For batch-processing a list of prompts, prompts can be specified from an input file via `-pf`/`--prompts-file`. Each line will be interpreted as one prompt, and prompts will be run sequentially.

### Image settings
- The flags `-W`/`--W` and `-H`/`--H` specify image resolution in width and height, respectively. Input values which are not divisible by 64 will be truncated to a multiple of 64 automatically.
### Diffusion settings
- `-S`/`--seed` will set the image seed. So long as the image size remains the same, keeping the seed should yield "the same image" under a similar prompt, or "a similar composition" under different prompts.
- `-s`/`--steps` will set the amount of diffusion steps performed. Higher values can help increase detail, but will be more computationally expensive.
- `-cs`/`--scale` sets the guidance scale. Increasing this value may make outputs more adherent to the prompt, while decreasing it may increase 'creativity'. The effect of different values will be different depending on the scheduler used.
- `-sc`/`--scheduler` sets the sampling scheduler, with `mdpms` being used by default. See [the huggingface diffusers list of implemented schedulers](https://huggingface.co/docs/diffusers/api/schedulers#implemented-schedulers) for more information. Currently, the following schedulers are available: <!--["lms", "pndm", "ddim", "euler", "euler_ancestral", "mdpms", "sdpms", "kdpm2", "kdpm2_ancestral", "heun"]-->
  - `"lms"`: LMSDiscrete
  - `"pndm"`: PNDM
  - `"ddim"`: DDIM
  - `"euler"`: EulerDiscrete
  - `"euler_ancestral"`: EulerAncestralDiscrete
  - `"mdpms"`: DPMSolverMultistep (dpmsolver++ algorithm, lower-order-final for \<15 timesteps)
  - `"sdpms"`: DPMSolverSinglestep (dpmsolver++ algorithm, lower-order-final)
  - `"kdpm2"`: KDPM2Discrete
  - `"kdpm2_ancestral"`: KDPM2AncestralDiscrete
  - `"heun"`: HeunDiscrete
- `-e`/`--ddim-eta` sets the eta (η) parameter when the ddim scheduler is selected. Otherwise, this parameter is ignored. Higher values of eta will increase the amount of additional noise applied during sampling. A value of `0` corresponds to no additional sampling noise.
- `-es`/`--ddim-eta-seed` sets the seed of the sampling noise when a ddim scheduler with eta > 0 is used.
- `-gsc`/`--gs-schedule` sets a schedule for variable guidance scale. This can help with mitigating potential visual artifacts and other issues caused by high guidance scales. By default (None), a static guidance scale with no schedule will be used. The schedule will be scaled across the amount of diffusion steps (`-s`), yielding a multiplier between `0` and `1` for the guidance scale specified via `-cs`.
  - Currently, the following schedules are available: `None`, `"sin"` (1/2-period sine between 0 and π: 0→1→0), `"cos"` (1/4-period cosine between 0 and π/2: 1→0), `"isin"` (inverted sin (1-sin): 1→0→1), `"icos"` (inverted cos (1-cos): 0→1), `"fsin"` (full-period sine between 0 and 2π: 0→1→0→-1→0), `"anneal5"` (2.5 periods of a rectified sine (abs(sin) between 0 and 5π), yielding 5 sequential "bumps" of 0→1→0), `"rand"` (random multiplier between 0 and 1 in each step), `"frand"` (random multiplier between -1 and 1 in each step)
### Device, Performace and Optimization settings
- `--unet-full` will switch from using a half precision (fp16) UNET to using a full precision (fp32) UNET. This will increase memory usage significantly. See section [Precision](#Precision).
- `--latents-half` will switch from using full precision (fp32) latents to using half precision (fp16) latents. The difference in memory usage should be insignificant (<1MB). See section [Precision](#Precision).
- `--diff-device` sets the device used for the UNET and diffusion sampling loop. `"cuda"` by default.
- `--io-device` sets the device used for anything outside of the diffusion sampling loop. This will be text encoding and image decoding/encoding. `"cpu"` by default. Switching this to `"cuda"` will increase VRAM usage (see the example shown in section [Precision](#Precision)), while only speeding up the (significantly less time intensive!) encode and decode operations before and after the sampling loop.
- `--seq`/`--sequential_samples` will process batch items (if multiple images are to be generated) sequentially, instead of as a single large batch. Reduces VRAM consumption. This flag will activate automatically if generating runs out of memory when more than one image is requested.
- `-as`/`--attention-slice` sets slice size for UNET attention slicing, reducing memory usage. The value must be a valid divisor of the UNET head count. Set to 1 to maximise memory efficiency. Set to 0 to use the diffusers recommended "auto"-tradeoff between memory reduction and (minimal) speed cost.
- `-co`/`--cpu-offload` will enable CPU offloading of models through `accelerate`. This should enable compatibility with minimal VRAM at the cost of speed.
- `-cb`/`--cuda-benchmark` will perform CUDNN performance autotuning (CUDA benchmark). This should improve throughput when computing on CUDA, but will have a slight overhead and may slightly increase VRAM usage.
#
## image to image
When either a starting image (`-ii`, see below) or the image cycle flag (`-ic`, see below) are specified, `generate.py` automatically performs image-to-image generation.
### Image to image flags
- `-ii`/`--init-image` specifies the path to a starting image file. The image will automatically be scaled to the requested image size (`-H`/`-W`) before image-to-image is performed.
- `-st`/`--strength` specifies the 'strength' setting of image-to-image. Values closer to `0` will result in the output being closer to the input image, while values closer to `1` will result in the output being closer to the information of the newly generated image.
### Image to image cycling
When applying image-to-image multiple times sequentially (often with a lower strength), the resulting outputs can often be "closer" to the input image, while also achieving a higher "image quality" or "adherence to the prompt". Image-to-image cycling can also be applied without a starting image (the first image in the sequence will be generated via text-to-image) to create interesting effects.

For faster generation cycles, it is recommended to pass the flags `--no-check-nsfw` (you will not receive warnings about potential NSFW content being detected, see: [Additional Flags](#additional-flags)) and `--io-device cuda` (VRAM usage will be increased, see: [Device settings](#device-settings))
- `-ic`/`--image-cycles` sets the amount of image-to-image cycles that will be performed. An animation (and image grid) will be stored in the `/animated` folder in the output directory. While running, this mode will attempt to display the current image via cv2. This can be disabled by setting the global variable `IMAGE_CYCLE_DISPLAY_CV2=False` near the top of `generate.py`.
  - When multiple prompts are specified via `||` (see: [Multiple Prompts](#multiple-prompts)), an interpolation through the prompt sequence will be performed in the text encoder space.
    - Unlike in earlier versions of this repository, this now works for interpolating between complex prompts which combine/mix multiple subprompts with their own internal prompt weights (see: `;;` in [Usage & Features](#usage--features))
- `-cni`/`--cycle-no-save-individual` disables the saving of image-to-image cycle frames as individual images when specified.
- `-iz`/`--image-zoom` sets the amount of zoom applied between image-to-image steps. The value specifies the amount of pixels cropped per side. Disabled with a value of `0` by default.
- `-ir`/`--image-rotate` sets the amount of degrees of (counter-clockwise) rotation applied between image-to-image steps. Disabled with a value of `0` by default.
- `-it`/`--image-translate` sets the amount of translation applied to the image between image-to-image steps. This requires two values to be specified for the x and y axis translation respectively. Disabled with a value of `None` by default.
- `-irc`/`--image-rotation-center` sets the position of the rotational axis within the image in pixel coordinates, if rotations are applied. Requires two values for both x and y coordinates, with the origin `0,0` being the top left corner of the image. By default, this automatically selects the center of the image with a value of `None`.
- `-ics`/`--image-cycle-sharpen` sets the strength of the sharpening filter applied when zooming and/or rotating during image-to-image cycling. This filter is only applied to image inputs before the next cycle, not to stored image outputs. This can help preserve image sharpness, as the resampling applied when zooming or rotating will soften or blur the image. Values greater than `1.0` will increase sharpness, while values between `1.0` and `0.0` will soften the image. Default is `1.2`.
- `-icc`/`--image-color-correction` Enables color correction for image to image cycling: Cumulative density functions of each image channel within the LAB colorspace are respaced to match the density distributions present in the initial (first) image. Prevents 'magenta shifting' (and similar effects) with multiple cycles.
#
## Additional flags
- `-om`/`--online-model` can be used to specify an online model id for acquisition from huggingface hub. This will override the default local (manual) and automatic models. See: [Automatic model install](#option-a-automatic-model-install-via-huggingface)
- `-lm`/`--local-model` can be used to specify a directory containing local model files. This directory should contain `unet` and `vae` dirs, with a `config.json` and `diffusion_pytorch_model.bin` file each. See: [Manual model install](#option-b-manual-model-install)
- `-od`/`--output-dir` sets an override for the base output directory. The directory will be created if it is not already present.
- `--no-check-nsfw` disables the NSFW check entirely, which slightly speeds up the generation process. By default, `generate.py` will only display a warning and attach an extra tag in image metadata if potential NSFW concepts are detected.
- `--animate` will store any intermediate (unfinished) latents during the sampling process in CPU memory. After sampling has concluded, an animation (and image grid) will be created in the `/animated` folder in the output directory
- `-in`/`--interpolate-latents` accepts two image paths for retrieving and interpolating latents from the images. This will only work for images of the same size which have had their latents stored in metadata (`generate.py` does this by default, as it will only increase image size by 50-100kB). While the interpolation occurs in the latent space (after which the VAE is applied to decode individual images), results will usually not differ from crossfading the images in image space directly. Results are saved like in `--animate`.
- `-cfi`/`--cycle-fresh-image` when combined with image cycles (`-ic`), a new image will be created via text-to-image for each cycle. Can be used to interpolate between prompts purely in text-to-image mode (fixed seed recommended).
- `-mn`/`--mix-negative-prompts` switches to mixing negative prompts directly into the prompt itself, instead of using them as uncond embeddings. See [Usage & Features](#usage--features)
- `-dnp`/`--default-negative-prompt` can be used to specify a default negative prompt, which will be utilized whenever no negative prompts is given.
- `-cls`/`--clip-layer-skip` can be used to specify a default CLIP (text encoder) skip value if none is specified in the prompt. See [Usage & Features](#usage--features)


# Precision
When switching the precision of either the unet or the latents between full (fp32) and half (fp16), there will be a small difference in outputs.

(Measurements were taken under an older, less optimised version of diffusers. Exact values may no longer be accurate.)
Switching from a half precision UNET to a full precision UNET will significantly increase the amount of memory required: ~9.15GB instead of ~6.45GB (overall usage) in the example shown below.
Switching from full precision latents to half precision latents should only reduce the memory usage by less than a megabyte.

The following examples all use the prompt "test one two" with a fixed seed of `-S 1`. The results are (in order):
- half(fp16) UNET and full(fp32) latents [`default`]
- half(fp16) UNET and half(fp16) latents [`--latents-half`]
- full(fp32) UNET and full(fp32) latents [`--unet-full`]
  
<img src="./examples/fp16_unet_fp32_latents.png" width="30%" height=25%/> <img src="./examples/fp16_unet_fp16_latents.png" width="30%" height=25%/> <img src="./examples/fp32_unet_fp32_latents.png" width="30%" height=25%/>

Visualization of the difference between outputs in this example (highlighting of pixels where the differences are greater than a 1% fuzz threshold):
- half(fp16) latents compared to full(fp32) latents (with UNET at half(fp16) precision)
- half(fp16) UNET compared to full(fp32) UNET (with latents at full(fp32) precision)

<img src="./examples/compare_latent_precision.png" width="30%" height=25%/> <img src="./examples/compare_unet_precision.png" width="30%" height=25%/>

#
# Discord bot
**The included discord bot script is provided AS IS.** It is primarily intended to serve as an example, or as a starting point for custom bot scripts.
- `discord_bot.py` contains an example of a discord bot script, which imports functions from `generate.py` to perform both image-to-image and text-to-image generation. This example bot is not recommended for use outside of trusted, private servers in channels marked as NSFW.
- Outputs flagged as potentially NSFW will be sent as spoilers, however, this will likely not make them exempt from any content filters or rules.
- This script uses two threads, one for the discord bot itself, and one for running StableDiffusion. They communicate via queues of pending and completed tasks.
- Some basic settings (saving outputs to disk, precision, devices, command prefix) can be set using global variables at the top of the script:
  - `SAVE_OUTPUTS_TO_DISK` can be set to `False` if outputs generated through bot commands should not be saved on the device running the bot.
  - `DEFAULT_HALF_PRECISION` can be set to `False` if the fp32 version of the UNET should be used
  - `IO_DEVICE` can be set to `"cpu"` if running in a more memory optimized (slightly slower) configuration is desired.
  - `RUN_ALL_IMAGES_INDIVIDUAL` can be set to `True` to default to running batch items in sequence. Otherwise, the automatic failover should trigger in response to oversized batches.
  - `FLAG_POTENTIAL_NSFW` can be set to `False` to entirely disable the content check. This will speed up generation slightly, while no longer flagging images and marking them as spoilers.
  - `USE_HALF_LATENTS` see: `--latents-half` in [Device, Performace and Optimization settings](#device-performace-and-optimization-settings)
  - `ATTENTION_SLICING` see: `-as` in [Device, Performace and Optimization settings](#device-performace-and-optimization-settings)
  - `CPU_OFFLOAD` see: `-co` in [Device, Performace and Optimization settings](#device-performace-and-optimization-settings)
  - `PERMIT_RELOAD` can be set to `True` to allow users to switch the current model, toggle CPU offloading and set attention slicing via the `/reload` command.
    - If set to `True`, `permittel_local_model_paths` specifies a whitelist of local model names with their respective model paths (see: [Manual model install](#option-b-manual-model-install)), while `permitted_model_ids` specifies a whitelist of names with respective huggingface hub model ids (see: [Automatic model install](#option-a-automatic-model-install-via-huggingface))
- Available commands are specified via discord `slash commands`. The pre-existing commands serve as a starting point for creating optimal commands for your use-case.
- The bot utilizes the automatic switching to image-to-image present in `generate.py`. When a valid image attachment is present, it will be used as the input for image-to-image.
- In case of an error, the bot should respond with an error message, and should continue to function.

## Installation
In addition to the dependencies used for `generate.py` (see [Installation](#installation)), the bot script requires [`py-cord`](https://github.com/Pycord-Development/pycord), a fork and continuation of `discord.py`. It can be installed via:
```shell
pip install -U py-cord
```
If `discord.py` is already installed in the environment, it will need to first be uninstalled before installing `py-cord` (both use the same `discord` namespace): 
```shell
pip uninstall discord.py
pip install py-cord
```

## Usage
- Set the bot token in the `tokens.py` file: `DISCORD_TOKEN = "your token here"`.
- Start `discord_bot.py`. The bot should now be accessible to anyone with access to whichever channels the bot is present in.
- The bot includes the following example commands (using discord slash commands):
  - `/square <text prompt> <Image attachment>` generates a default image with 512x512 resolution (overridden to 768x768 for SD2.x). Accepts an optional image attachment for performing image-to-image.
  - `/portrait <text prompt> <Image attachment>` (shortcut for 512x768 resolution images)
  - `/landscape <text prompt> <Image attachment>"` (shortcut for 768x512 resolution images)
  - `/advanced <text prompt> <width> <height> <seed> <guidance_scale> <steps> <img2img_strength> <Image attachment> <amount> <scheduler> <gs_schedule> <ddim_eta> <eta_seed>`
    - `Width` and `height` are specified either as pixels (for values >64), or as a multiplier of 64, offset from 512x512. A `width` of `3` and `height` of `-2` will result in an image which is `512+64*3 = 704` pixels wide and `512-64*2 = 384` pixels high
    - If seeds are set to a value below `0`, the seed is randomized. The randomly picked seed will be returned in the image response.
    - `scheduler` and `gs_schedule` display available options.
    - Unless a source image is attached, `img2img_strength` is ignored.
    - Steps are limited to `150` by default.
  - `/reload <model name> <enable cpu offload> <attention slicing> <default CLIP skip>` if `PERMIT_RELOAD` is changed to True, this can be used to (re-)load the model from a selection of available models (see above).
  - `/default_negative <negative prompt>` can be used to set a default negative prompt (see: `-dnp`, [Additional flags](#additional-flags)). If <negative_prompt> is not specified, the default negative prompt will be reset.
- All commands come with a short documentation of their available parameters.

#
# Notes
- For more information about [Stable Diffusion](https://github.com/CompVis/stable-diffusion) and the [huggingface diffusers model of Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion-v1-4), including the license, limitations, and capabilities of the systems utilized, check out the respective links.
- Text to image and image to image implementations are derived from the pipeline implementations of the [huggingface diffusers library](https://github.com/huggingface/diffusers)
