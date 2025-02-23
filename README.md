# Yet Another StableDiffusion Implementation
Stable Diffusion script(s) based on huggingface diffusers. Comes with extra configurability and some bonus features, a single script for accessing every functionality, and example code for a discord bot demonstrating how it can be imported in other scripts.
Extra scripts for model acquisition or conversion (converting from monolithic single-file models to diffusers files) are also included.
## Recent changes to requirements
- Updating `diffusers`, `transformers`, `huggingface-hub`, `accelerate`, `pytorch` and `xformers` (if installed) is generally recommended.
- controlnet_aux can be installed for additional controlnet preprocessors. Users of previous versions should upgrade to >= 0.0.8
- Upgrading Python from older versions to Python 3.9 is now required.
- xformers can optionally be installed to potentially boost efficiency (testing required for Pytorch2.0+, effectiveness may vary depending on hardware)
  - If xformers is not installed, this implementation should have Diffusers default to [PyTorch2.0 efficient attention](https://huggingface.co/docs/diffusers/optimization/torch2.0). This should perform similarly to xformers efficient attention. Especially without using PyTorch2.0 model compilation (not platform-agnostic), using xformers may still provide slightly better performance.
  - The latest version of xformers often **requires** a specific (usually the latest) version of pytorch. Installing xformers with an older (or newer) version of pytorch may update pytorch to a cpu-only version. [Manually updating pytorch](#install-pytorch-skip-when-using-a-pre-existing-stablediffusion-compatible-environment) to the correct version with CUDA support may be required.
  - Depending on the CUDA capability of your hardware, you may need to manually specify the correct index URL when installing. See installation information on the [installation section of the xformers repository](https://github.com/facebookresearch/xformers#installing-xformers).
- [lora_diffusion](https://github.com/cloneofsimo/lora.git) can be installed to load their compatible LoRA embeddings.
- The `scikit-image` package is required when performing color correction during image cycling


# Python Installation
Python version 3.9+ is required. Testing is currently carried out under Python 3.9.
### Install `pytorch`
- Get the correct installation for your hardware at https://pytorch.org/get-started/locally/.
  - When installing pytorch with CUDA support, the `conda` install command will install cudatoolkit as well.
  - When installing with `pip` directly (e.g. for non-conda environments), CUDA toolkit may also need to be manually installed. CUDA toolkit can be found at: https://developer.nvidia.com/cuda-downloads.
### Install additional dependencies
```shell
pip install --upgrade diffusers transformers scipy ftfy opencv-python huggingface_hub scikit-image accelerate controlnet_aux
pip install git+https://github.com/cloneofsimo/lora.git
```

> [!IMPORTANT]
> ### GPU acceleration on windows without CUDA
> Experimental support for the [DirectML](https://github.com/microsoft/DirectML) backend (using [torch-directml](https://pypi.org/project/torch-directml/)) should allow windows users to use GPU acceleration on any DirectX 12 capable device without native pytorch support. (This should permit GPU acceleration on e.g. AMD Radeon GPUs)
>
> To use directML, install the `torch-directml` library: `pip install torch-directml`
>
> When CUDA is unavailable and the `torch-directml` library is detected, it should automatically be set as the default device. When specifying directML as a backend manually, use `"dml"` or `"directml"` as the device name.
>
> Note that some features may be broken. In preliminary testing, certain configurations were able to cause the `generate.py` to freeze entirely, neither crashing, nor proceeding. The process may need to be manually terminated when frozen. Known issues include:
> - Generation can freeze at the text encoder stage when a LoRA is loaded and CPU offload is applied.

#
# Model Installation

### Specifying a remote model from huggingface.co (SD 1.x only):
<details><summary>[Loading a huggingface model from a model ID]</summary>

- You should have read and accepted the license terms ([CreativeML OpenRAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license)) of the relevant StableDiffusion model repository at https://huggingface.co/CompVis/stable-diffusion-v1-4.
- Model installation should occur automatically, even without being logged into huggingface hub. A custom model id can be specified via `-om` (see:[Specifying models](#specifying-models))
- If your (custom) online model repository requires access credentials, you can provide them via the following:
  - Either paste the token into the `tokens.py` file: `HUGGINGFACE_TOKEN = "your token here"`
  - Or log into huggingface-cli with your token (run `huggingface-cli login` in a terminal while your python environment is activated). This login should remain stored in your user directory until you log out with the cli, independent of your python environment.
</details>

## Local model installation:
### Converting monolithic models (single model file: `.safetensors` or `.ckpt`/`.pt`/...)
Most models, especially custom models, tend to be distributed as a single file, intended for use with the 'standard' Stablediffusion implementation. Frequently, no native files for diffusers are available.
`convert_model_files.py` Can be used to convert a single model file of any custom StableDiffusion model into the relevant diffusers files:
```shell
# this will load the model file './my_custom_model.safetensors' and create a diffusers model in 'models/my_custom_model'
python convert_model_files.py my_custom_model.safetensors
# optionally, a different name for the diffusers model can be specified: this will create a diffusers model in 'models/myModelName'
python convert_model_files.py my_custom_model.safetensors --name myModelName
```
- The script should function on both 'standard' StableDiffusion models (SD1.x, untested for SD2.x) and SDXL models (this includes models derived from Playground). Refiner models are currently not supported
- Converting file formats outside of `.safetensors` (usually `.ckpt`) is possible, but should only be done if you are certain that you can trust the author. Non-`.safetensors` model files are capable of containing arbitrary data, including malicious code.

### Downloading a model via a model ID from [huggingface](https://huggingface.co/models)
To automatically download a model from https://huggingface.co while avoiding any unneeded files (e.g. model files for different implementations/backends), `download_model_files.py` can be used:
```shell
# download StableDiffusion v1.5 to 'models/SDv1.5'
python download_model_files.py runwayml/stable-diffusion-v1-5 --name SDv1.5
# download Playground's 'Playground v2 1024px aesthetic' model to 'models/playgroundv2_aesthetic'
python download_model_files.py playgroundai/playground-v2-1024px-aesthetic --name playgroundv2_aesthetic
# download StableDiffusionXL-base-v1.0 to 'models/stable-diffusion-xl-base-1.0' (name inferred from the model ID as none was specified)
python download_model_files.py stabilityai/stable-diffusion-xl-base-1.0
# download StableDiffusion v2.1 to 'models/stable-diffusion-2-1' (name inferred)
python download_model_files.py stabilityai/stable-diffusion-2-1
# download Playground v2.5
python download_model_files.py playgroundai/playground-v2.5-1024px-aesthetic --name playgroundv2-5_aesthetic
```
- The script should function on 'standard' StableDiffusion models (SD1.x and SD2.x) as well as SDXL models or other models with an analogous architecture (e.g. [Playground v2](https://huggingface.co/playgroundai/playground-v2-1024px-aesthetic))
- Downloads are performed by `huggingface_hub`. If supported, this will download larger model files to a user cache directory and place symbolic links in the model directory to avoid storing multiple copies. This behavior can be disabled via the `--no-link` flag.
- `--name` can be used to specify the local name of the model (folder name in `models`). If not specified, the remote model name will be used.
- Some models may not provide `.safetensors` files, only having `.pt` model files available. The `--pickle` flag can be used to download such files instead. Only use these files if you are certain that you can trust the author. Non-`.safetensors` model files are capable of containing arbitrary data, including malicious code.
  - If such a model is also available as a single `.safetensors` file, it is recommended to download and [convert this file](#converting-monolithic-models-single-model-file-safetensors-or-ckptpt) instead.
- Some models may be mixed between `.safetensors` and `.pt` files. This could be the case if the model author provided e.g. the unet and vae files directly, with the text encoder being copied over from a different model, in a different format. To download both `.safetensors` and `.pt` files, the flag `-add-pickle` can be used.
  - If some model files are provided in both formats, this will download superfluous `.pt` files.
  - As noted above, downloading `.pt` files is not recommended if avoidable, and converting a single `.safetensors` file should be preferred if available.
- This should download all necessary files of a model. If a model uses different filenames, however, files may fail to be caught. The `--full` flag will enforce the downloading of all files provided in the repository.

### Other installation methods

#### Manually downloading files (not recommended)
<details><summary>[Example: Manually installing SD1.4, minimal SD1.x directory structure]</summary>

- Navigate to https://huggingface.co/CompVis/stable-diffusion-v1-4/. Note the license terms ([CreativeML OpenRAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license)).
- Head to the `Files and versions` tab, and navigate to `stable-diffusion-v1-4/unet`
  - Download both `config.json` and `diffusion_pytorch_model.bin` (or `diffusion_pytorch_model.safetensors`)
  - place both files in `models/stable-diffusion-v1-4/unet`
- Repeat for the VAE model: In the `Files and versions` tab, navigate to `stable-diffusion-v1-4/vae`
  - Download both `config.json` and `diffusion_pytorch_model.bin` (or `diffusion_pytorch_model.safetensors`)
  - Place the files in `models/stable-diffusion-v1-4/vae`
- Consider keeping the branch on the default: `main` instead of switching to `fp16`. The fp32 weights are larger (3.44GB instead of 1.72GB for SD1.4), but can be loaded for both full precision (fp32) and half precision (fp16) use.
- Note: The checkpoint files for diffusers are not the same as standard StableDiffusion checkpoint files (e.g. sd-v1-4.ckpt). They can not be copied over directly. See [Converting monolithic models](#converting-monolithic-models-single-model-file-safetensors-or-ckptpt)
- Note: The VAE files can instead be replaced by an StabilityAIs improved `ft-EMA` / `ft-MSE` VAEs, which are available under [huggingface.co/stabilityai/sd-vae-ft-mse/tree/main](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main)

The following files should be present in the model directory. `.bin` files may be substituted with `.safetensors` files. Additional files may be present, especially when the model is acquired via `git clone`.
```shell
v1.5/
├── unet/
│   ├── config.json
│   └── diffusion_pytorch_model.bin
└── vae/
    ├── config.json
    └── diffusion_pytorch_model.bin
```
</details>

#### Local installation via git
<details><summary>[Example: installing StableDiffusion v2.1 via git lfs]</summary>

Git must be installed on your system.
Note the license terms ([CreativeML OpenRAIL++-M](https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/LICENSE-MODEL)).

From the root directory of this repository, clone your model repository into `models/` (this example will clone StableDiffusion v2.1):
```shell
git lfs clone https://huggingface.co/stabilityai/stable-diffusion-2-1 models/stable-diffusion-2-1 --progress -X "v2-1_768-ema-pruned.ckpt","v2-1_768-nonema-pruned.ckpt"
```
This may take a while.
Using 'git lfs clone' will yield a deprecation warning, however, cloning through 'git clone' may fail to display any progress information on some platforms.
</details>

#### A Note on StableDiffusion version 2.x+:
<details><summary>[SD 2.x model installation notes, minimal SD 2.x directory structure]</summary>

Currently, only [Local model installation](#local-model-installation) is supported for SD2.x models. StableDiffusion v2.1 can be acquired from https://huggingface.co/stabilityai/stable-diffusion-2-1.
In addition to the `unet` and `vae` folders, the `scheduler`, `text_encoder` and `tokenizer` folders must also be added to the model directory, together with their respective files:
- `models/*/scheduler/`: `scheduler_config.json`
- `models/*/text_encoder/`: `config.json` and `pytorch_model.bin` (or `pytorch_model.safetensors`)
- `models/*/tokenizer/`: `merges.txt`, `special_tokens_map.json`, `tokenizer_config.json` and `vocab.json`

If `scheduler/scheduler_config.json` is not provided, the model will be presumed to not be a v_prediction model (this will cause issues with anything but the _base_ variant of SD2.x).
If `text_encoder` and `tokenizer` do not provide the required files, the model will be loaded with `openai/clip-vit-large-patch14`, which is used in SD1.x-style models. This is incompatible with SD2.x.

The following files should be present in the model directory. `.bin` files may be substituted with `.safetensors` files. Additional files may be present, especially when the model is acquired via `git clone`.
```shell
stable-diffusion-2-1/
├── scheduler/
│   └── scheduler_config.json
├── text_encoder/
│   ├── config.json
│   └── pytorch_model.bin
├── tokenizer/
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── unet/
│   ├── config.json
│   └── diffusion_pytorch_model.bin
└── vae/
    ├── config.json
    └── diffusion_pytorch_model.bin
```
</details>

#
# Usage & Features
- StableDiffusion is supported for version 1.x (e.g. SD v1.5), version 2.x (e.g. SD v2.1), and StableDiffusionXL, including custom models. This includes support for [Playground v2](https://huggingface.co/playgroundai/playground-v2-1024px-aesthetic).
  - Refiner models are not currently supported. For SDXL, only the 'base' models can be used.
- Features include text-to-image, image-to-image (including cycling), and options for precision/devices to control generation speed and memory usage.
  - Advanced prompt manipulation options including mixing, concatenation, custom in-prompt weights and (per-token) CLIP-skip settings are available.
  - Additional memory optimisation options include (automatic) sequential batching, the usage of xformers attention by default, attention/vae slicing and CPU offloading (see: [Optimisation settings](#device-performace-and-optimization-settings))
- Animating prompt interpolations is possible for both image-to-image cycling and text-to-image (`-cfi`, see: [Additional Flags](#additional-flags)).
- Cycling through iterations of image-to-image comes with a mitigation for keeping the color scheme unchanged, preventing effects such as 'magenta shifting'. (`-icc`, see: [Image to image cycling](#image-to-image-cycling))
- For every generated output, an image will be saved in `outputs/generated`. When multiple images are batched at once, this singular image will be a grid, with the individual images being saved in `outputs/generated/individual`. Additionally, each generated output will produce a text file in `outputs/generated/unpub` of the same filename, containing the prompt and additional arguments. This can be used for maintaining an external image gallery.
  - Optionally, a different output directory can be specified (`-od`, see: [Additional flags](#additional-flags)).
  - `prompts`, `seeds`, and other relevant arguments will be stored in PNG metadata by default.
- The NSFW check will (by default) attach a metadata label, and will perform a local blur corresponding to detected category labels. The processing level may be set (or fully disabled, although this is not recommended) via commandline flags (see: [Additional Flags](#additional-flags)). Note that any biases, false negatives, or false positives caused by CLIP can adversely affect processing results.
  - When the processing performs local blurring, the blur mask is generated via [CLIPSeg](https://huggingface.co/blog/clipseg-zero-shot) for any labels detected by the default safety checker. As absolute detection levels of CLIPSeg are likely less reliable than those produced by the safety checker (CLIPSeg is based on a smaller, less accurate CLIP model), and are thus far uncalibrated for safety checking purposes, the blur mask is generated by selecting any areas of high relative intensity for the given labels. If the safety checker detects a label for which CLIPSeg is not able to provide a sufficiently confident segmentation, this will result in more of the image being obscured.
### Extensions
- "Textual Inversion Concepts", custom prompt embeddings, from https://huggingface.co/sd-concepts-library can be placed in `models/concepts` (only the .bin file is required, other files are ignored). They will be loaded into the text encoder automatically. `.pt`-style custom embeddings are also supported in the same way.
  - When using an SDXL model (or any other model with multiple text encoders), concept embeddings will be loaded into any text encoder with a matching feature dimension. This allows embeddings created for SD1.x to be used on the first of the two text encoders of SDXL.
- LoRA embeddings are supported (`-lop`,`-low`, see: [Additional Flags](#additional-flags)). This includes native diffusers attn_procs LoRA embeddings, ["lora_diffusion"](https://github.com/cloneofsimo/lora.git) embeddings and other convertable embeddings (LoRA embeddings made/compatible with ["kohya-ss/sd-scripts"](https://github.com/kohya-ss/sd-scripts), with both 'normal' linear LoRAs and convolutional LoRAs being supported).
  - The integrated converter should be able to load LoCon and experimental IA3 '(IA)^3' [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) models. Requests for other useful types of "LoRA-like" extension models are welcomed in the issues section.
  - The LoRA converter is derived and modified from the [haofanwang/diffusers](https://github.com/haofanwang/diffusers/blob/75501a37157da4968291a7929bb8cb374eb57f22/scripts/convert_lora_safetensor_to_diffusers.py) conversion script, see [diffusers PR#2403](https://github.com/huggingface/diffusers/pull/2403)
- ControlNets are supported, with included image preprocessors and dynamic (scheduled) ControlNet strength/guidance scale. Refer to the [ControlNet](#controlnet) flags section for details.
  - If displaying freshly generated images with cv2 is enabled, an additional preview window will be created when preprocessing ControlNet inputs by default. This can be used to inspect the 'correctness' of the preprocessed image.
### Prompt manipulation
- Prompts with a dynamic length (no token limit) now use an advanced moving window encoding method to minimize loss of information/context across the prompt.
  - A window the size of the encoder token limit is moved along the prompt length, encoding the prompt at every possible offset until a window containing the final token is reached. The different CLIP-vector representations obtained for each token are then mixed to obtain the representation which is passed on to the generator.
  - When multiple representations for one token were derived, they are mixed using weights given by a gaussian distribution which prioritises representation vectors closer to the center of their encoding window (more surrounding information on both sides of the token). The global variable `WINDOW_ENCODING_REDUCE_SIGMA` near the top of `generate.py` currently controls the spread (std) of these gaussian weights. Higher values mix representations closer to their window edges more equally, while 0 simply picks the representation vector closest to the respective encoder window center. When set to None, the highest magnitude value for each dimension (index of absolute max) across all representations is used.
  - Note that this method will require significantly more evaluations of the text encoder for longer prompts (`n - token_limit` evaluations for n tokens compared to `n / token_limit` evaluations for the previous chunking method). However, unless text encoding is performed on a slower device (e.g. encoding prompts on CPU when running the diffusion on GPU), even encoding very long prompts of a few hundred tokens should not make up the majority of generation time.
  - Switching back to the common (previous) method of handling long prompts (chunking, then encoding and re-concatenating) requires setting the global variable `USE_CHUNKED_LONG_PROMPTS = True` near the top of `generate.py`. It should be noted that supporting prompts with no length limit in this way is flawed, as the text encoder will not be able to consider information from adjacent prompt chunks for context when encoding. It may be preferable to manually chunk prompts into self-contained sub-prompts instead of natively using chunk-based long-prompt handing (see usage of the concatenation mixing mode below, enable via `-mc`, [Additional Flags](#additional-flags))
- Prompts can be mixed using prompt weights: When prompts are separated by `;;`, their representation within the text encoder space will be averaged. Custom prompt weights can be set by putting a number between the `;;` trailing the prompt. If no value is specified, the default value of 1 will be used as the prompt weight. 
  - Example: `"Painting of a cat ;3; Photograph of a cat ;1;"` Will yield the text representation of `3/4 * "Painting of a cat" + 1/4 "Photograph of a cat"`
  - By default, all negative subprompts will be mixed according to their weight, and used in place of the unconditional embedding for classifier-free guidance (standard "negative prompts"). In this case, the difference in weights between the positive and negative prompts is not considered, as this is given by the guidance scale (`-cs`).
  - Alternatively, prompts with negative weight values can be directly mixed into the prompt itself, leaving the unconditional embedding untouched (`-mn`, [Additional Flags](#additional-flags)). In this case, negative prompts are not directly 'subtracted'. Instead, the prompt is amplified in its difference from the negative prompts (moving away from the prompt, in the opposite direction of the negative prompt). Relative weight values between positive and negative prompts are considered. This way of applying negative prompts tends to be far more chaotic, but can yield interesting results. In this mode, a loose list of unwanted attributes as a negative prompt will usually perform worse than a description of the desired image together with negative attributes.
  - When switching to 'concatenation' mixing mode (`-mc`, see: [Additional Flags](#additional-flags)), mixed prompts have their embeddings multiplied by their prompt weight, and are then concatenated into one (longer) prompt. This will make use of dynamic length support where necessary. Additionally, a `+` can be appended to a prompt weight to signify that the end token embedding of the preceding prompt and the start token embedding of the subsequent prompt should be removed, resulting in a 'more direct' concatenation. This is interpreted as a 'normal concatenation' (instead of applying a sum) when not running in concatenation mode. (Example: `Painting of a cat;1.25+;Photograph of a cat;0.8;`)
- Prompts can be specified with an (individual) CLIP-skip setting by appending a trailing `{cls<n>}` for a setting of `n`. This will skip the last *n* layers of CLIP, the text encoder. Increasing this value will reduce the "amount of processing"/"depth of interpretation" performed by the text encoder. A value of `0` is equivalent to specifying no CLIP-skip setting.
  - **Note:** Other StableDiffusion implementations may refer to disabled CLIP-skipping as 'a CLIP skip of 1', not 'a CLIP skip of 0'. In this case, equivalent CLIP-skip values in this implementation will always be one less.
  - **Note:** StableDiffusionXL and similar models override a CLIP-skip of 0 to a CLIP-skip of 1 by default. The pooled prompt ignores any CLIP-skip setting. To experiment with a CLIP-skip of 0 on SDXL prompts it can be specified via `{cls0}` or via in-prompt weights `( :1;0)`.
  - Example: `"Painting of a cat{cls2}"` will encode "Painting of a cat", while skipping the final two layers of the text encoder.
  - When combined with prompt mixing or negative prompts (`;;`, see above), the prompt separator must be specified after the CLIP-skip setting. The skip setting is independent for each sub-prompt.
    - Example: `"Painting of a cat{cls1};3; Photograph of a cat{cls2};1;"`
    - As shown in the example, this can also be used to mix text prompts with themselves under different skip settings, or for interpolating between the same prompt under different skip settings (see: [Cycling](#image-to-image-cycling) and `-cfi` under [Additional Flags](#additional-flags))
  - For some custom models, using a specific CLIP-skip setting by default is recommended. The default value used when none is specified can be set via `-cls` (see: [Additional Flags](#additional-flags)).
- Custom in-prompt weights are available through a syntax similar to the [lpw_stable_diffusion](https://huggingface.co/docs/diffusers/using-diffusers/custom_pipeline_examples#long-prompt-weighting-stable-diffusion) pipeline:
  - Prompt sections can be encased by `( :x)`, with `x` being the weight of the prompt section contained within the brackets. `x` accepts any floating point number, and can therefore be used to specify both an increase in weight (`x>1`) or a decrease in weight (`x<1`). Negative values can also be used to create interesting effects, however, this may often fail to provide a 'negative prompt'-style result and can sometimes reduce output quality.
  - Additionally, this syntax can be used to (optionally) specify various CLIP-skip levels (see above) for individual sections of a prompt. To do this, prompt sections can be encased by `( :x;n)`, with `x` being the section weight (any floating point number) and `n` being the CLIP-skip level of the section (any integer within the available range given by text encoder depth). Local CLIP-skip settings specified this way will temporarily override both the prompt-level CLIP-skip setting (`{cls<n>}`) and the global CLIP-skip setting (`-cls`).
  - Prompt weights are applied after the prompt encoding itself, and will not cause any (additional) fragmentation or chunking of the prompt. This is also the case for local CLIP-skip settings: The prompt is fully encoded, after which the embeddings for different CLIP-skip settings are interleaved according to the requested level on a per-token-basis/per-embedding-vector-basis.
  - Example: `"an (oil painting:0.8) of a cat, (displayed in a gallery:1.2;1)"` will decrease the magnitude of the embedding vectors encoding 'oil painting' by 20%, while utilizing the embedding vectors of the prompt with a CLIP-skip of 1 to encode 'displayed in a gallery', and increasing their magnitude by 20%. To only apply a local CLIP-skip without modifying prompt weights, a weight of 1 must be used: `"an oil painting of a cat, (displayed in a gallery:1;1)"`
  - Stacking multiple weight modifiers by encapsulating them inside eachother is not supported. Instead, individual 'effective' weights of sections must be specified in parallel.
- For models with multiple text encoders (SDXL or similar), different prompts can be specified for each text encoder by separating them with `//`.
  - This can be used to specify a Textual Inversion embedding (see: [Extensions](#extensions)) only in the prompt of the encoder that supports the embedding. Embeddings intended for use with SD1.x will be available on the first of the two encoders of SDXL.
  - This is evaluated after prompt separators (`||` and `;;`) but before any weight or skip specifiers (e.g. `( :1.5)`, `( :0.9;1)`, `{cls2}`)
    - Example (using every tweak possible): `"(Photograph:1.25) of a cat, pinned to the fridge {cls0}//Painting of a (cat:1.1;1), displayed in a (gallery :1;0){cls2};;<my_negative_embedding>//low quality, blurry;-1;"`
  - **Note:** The second text encoder of SDXL (or similar models) is likely to have a greater influence on the result, as it also provides the pooled prompt embedding vector. (Additionally, the second encoder provides a larger fraction of the standard prompt embedding, due to a larger feature dimension.)

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
- The flags `-W`/`--W` and `-H`/`--H` specify image resolution in width and height, respectively. Input values which are not divisible by 8 will be truncated to a multiple of 8 automatically.
### Diffusion settings
- `-S`/`--seed` will set the image seed. So long as the image size remains the same, keeping the seed should yield "the same image" under a similar prompt, or "a similar composition" under different prompts.
- `-s`/`--steps` will set the amount of diffusion steps performed. Higher values can help increase detail, but will be more computationally expensive.
- `-cs`/`--scale` sets the guidance scale. Increasing this value may make outputs more adherent to the prompt, while decreasing it may increase 'creativity'. The effect of different values will be different depending on the scheduler used.
- `-sc`/`--scheduler` sets the sampling scheduler, with `mdpms` being used by default. See [the huggingface diffusers list of implemented schedulers](https://huggingface.co/docs/diffusers/api/schedulers/overview) for more information. Currently, the following schedulers are available:
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
  - `"deis"`: DEISMultistepScheduler (lower-order-final for \<15 timesteps)
  - `"unipc"` : UniPCMultistepScheduler
  - `"tcd"`: TCDScheduler (primarily intended for use with a matching TCD LoRA model)
- `-ks`/`--karras-sigmas` specifies if the scheduler should use a 'Karras' sigma (σ) schedule if available, leading to progressively smaller sigma steps. `True` by default. This may sometimes be referred to as the "Karras" variant of a scheduler. See: [(arXiv:2206.00364)](https://arxiv.org/abs/2206.00364) (see `Eq. (5)`). The ρ (rho) parameter used should be 7 across all schedulers.
- `-e`/`--ddim-eta` sets the eta (η) parameter when the ddim scheduler is selected. Otherwise, this parameter is ignored. Higher values of eta will increase the amount of additional noise applied during sampling. A value of `0` corresponds to no additional sampling noise.
- `-es`/`--ddim-eta-seed` sets the seed of the sampling noise when a ddim scheduler with eta > 0 is used.
- `-gsc`/`--gs-schedule` sets a schedule for variable guidance scale. This can help with mitigating potential visual artifacts and other issues caused by high guidance scales. By default (None), a static guidance scale with no schedule will be used. The schedule will be scaled across the amount of diffusion steps (`-s`), yielding a multiplier (usually between `0` and `1`) for the guidance scale specified via `-cs`.
  - Currently, the following schedules are available: (Options are shared with scheduling [ControlNet](#controlnet) and [LoRA](#lora) strength, with some not being recommended for use on guidance scale.)
    - `None`: disable scheduling. Default.
    - `"sin"`: 1/2-period sine between 0 and π: 0→1→0
    - `"cos"`: 1/4-period cosine between 0 and π/2: 1→0
    - `"isin"`: inverted sin (1-sin): 1→0→1
    - `"icos"`: inverted cos (1-cos): 0→1
    - `"fsin"`: full-period sine between 0 and 2π: 0→1→0→-1→0
    - `"anneal5"`: 2.5 periods of a rectified sine (abs(sin) between 0 and 5π), yielding 5 sequential "bumps" of 0→1→0
    - `"ianneal5"`: inverted anneal5, 5 sequential "bumps" of 1→0→1
    - `"rand"`: random multiplier between 0 and 1 in each step
    - `"frand"`: random multiplier between -1 and 1 in each step
    - `"buzz"`: switching between 0 and 1
    - `"th2"`, `"th5"`, `"th7"`: (smoothly) switching from 1 to 0 at a threshold of respectively 25%, 50% and 75% of the overall steps.
    - `"ith2"`, `"ith5"`, `"ith7"`: (smoothly) switching from 0 to 1 at a threshold of respectively 25%, 50% and 75% of the overall steps.
- `-gr`/`--guidance-rescale` sets the guidance rescale factor 'φ' to improve image exposure distribution and (potentially _significantly_ increase adherence to both the prompt and training data). Disable with a value of 0. The authors of [the paper that introduces this correction (arxiv:2305.08891)](https://arxiv.org/abs/2305.08891) (see `(4)` and `5.2`) empirically recommend values between 0.5 and 0.75.
- `-fup`/ `--freeu-parameters` specifies FreeU parameters b1, b2, s1, s2 in-order. Disabled with a value of 0 (or 0 0 0 0) by default. Set -1 to automatically select the defaults recommended by the authors of this enhancement. See: [ChenyangSi/FreeU](https://github.com/ChenyangSi/FreeU)

### Device, Performace and Optimization settings
- `--unet-full` will switch from using a half precision (fp16) UNET to using a full precision (fp32) UNET. This will increase memory usage significantly.
- `--latents-half` will switch from using full precision (fp32) latents to using half precision (fp16) latents. The difference in memory usage should be insignificant (<1MB).
- `--diff-device` sets the device used for the UNET and diffusion sampling loop. `"cuda"` by default.
- `--io-device` sets the device used for anything outside of the diffusion sampling loop. This will be text encoding and image decoding/encoding. `"cuda"` by default. Switching this to `"cpu"` will decrease VRAM usage, while only slowing down the (significantly less time intensive!) encode and decode operations before and after the sampling loop.
- `--seq`/`--sequential_samples` will process batch items (if multiple images are to be generated) sequentially, instead of as a single large batch. Reduces VRAM consumption. This flag will activate automatically if generating runs out of memory when more than one image is requested.
- `-as`/`--attention-slice`
  - sets slice size for UNET attention slicing, reducing memory usage. The value must be a valid divisor of the UNET head count. Set to 1 to maximise memory efficiency. Set to 0 to use the diffusers recommended "auto"-tradeoff between memory reduction and (minimal) speed cost. This may be overridden by xformers attention if available and enabled.
  - additionally, if attention slicing is specified (with any value), the VAE will be set to always run in sliced and tiled mode, which should keep the memory requirements of VAE decodes constant between different image resolutions and batch sizes.
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

- `-ic`/`--image-cycles` sets the amount of image-to-image cycles that will be performed. An animation (and image grid) will be stored in the `/animated` folder in the output directory. While running, this mode will attempt to display the current image via cv2. This can be disabled by setting the global variable `IMAGE_CYCLE_DISPLAY_CV2=False` near the top of `generate.py`.
  - When multiple prompts are specified via `||` (see: [Multiple Prompts](#multiple-prompts)), an interpolation through the prompt sequence will be performed in the text encoder space.
    - Unlike in earlier versions of this repository, this now works for interpolating between complex prompts which combine/mix multiple subprompts with their own internal prompt weights (see: `;;` in [Usage & Features](#usage--features))
- `-cni`/`--cycle-no-save-individual` disables the saving of image-to-image cycle frames as individual images when specified.
- `-iz`/`--image-zoom` sets the amount of zoom applied between image-to-image steps. The value specifies the amount of pixels cropped per side. Disabled with a value of `0` by default.
- `-ir`/`--image-rotate` sets the amount of degrees of (counter-clockwise) rotation applied between image-to-image steps. Disabled with a value of `0` by default.
- `-it`/`--image-translate` sets the amount of translation applied to the image between image-to-image steps. This requires two values to be specified for the x and y axis translation respectively. Disabled with a value of `None` by default.
- `-irc`/`--image-rotation-center` sets the position of the rotational axis within the image in pixel coordinates, if rotations are applied. Requires two values for both x and y coordinates, with the origin `0,0` being the top left corner of the image. By default, this automatically selects the center of the image with a value of `None`.
- `-ics`/`--image-cycle-sharpen` sets the strength of the sharpening filter applied when zooming and/or rotating during image-to-image cycling. This filter is only applied to image inputs before the next cycle, not to stored image outputs.
  - Values greater than `1.0` will increase sharpness, while values between `1.0` and `0.0` will soften the image. Default is `1.0` with `1.2` being recommended when applying transforms (zoom/rotate).
  - This can help preserve image sharpness when applying transformations, as the resampling applied when zooming or rotating will soften or blur the image.
  - Applying a slight softening between cycles can help with increasingly sharpened outputs, and may prevent specific image details from becoming 'stuck'.
- `-icc`/`--image-color-correction` Enables color correction for image to image cycling: Cumulative density functions of each image channel within the LAB colorspace are respaced to match the density distributions present in the initial (first) image. Prevents 'magenta shifting' (and similar effects) with multiple cycles.
- `-csi`/`--cycle-select-image` When cycling images with batch sizes >1 in image to image mode (and with cv2 display enabled), pause to have the user manually select one of the outputs as the next input image after each cycle. This could be used to manually guide an 'evolution' of an image via a repeated, selective application of image to image.
  - To select the preferred image, simply click one of the images in the cv2 output window. Usability may be affected especially for larger batch sizes, as the display window created by `cv2.imshow` does not support adaptive window sizes or scrolling.
#
## Additional flags
- `-od`/`--output-dir` sets an override for the base output directory. The directory will be created if it is not already present.
- `-spl`/`--safety-processing-level` configures the level of content safety processing.
  - Levels 7 and 6 function like the diffusers pipeline default safety checking, either fully deleting the image (7), or coarsely pixelating/blurring the image (6).
  - Levels 5, 4, and 3 apply a local blur based on detected labels, selecting the following setting pairs for potential NSFW content / potential NSFW content with detected special labels (in order): boosted local blur (with additional detection margins) / full blur (5), local blur / full blur (4), local blur / boosted local blur (3).
  - Levels 2 and 1 will not process images unless special labels are detected. In case of special label presence, either boosted local blur (2) or local blur (1) is selected.
  - Level 0 will disable image safety processing entirely. THIS IS GENERALLY NEVER RECOMMENDED.
- `--no-check-nsfw` disables the NSFW check entirely, which very slightly speeds up the generation process. This disables both the metadata label and any post-processing associated with detecting potential NSFW content. THIS IS GENERALLY NEVER RECOMMENDED unless further content evaluation is performed downstream.
- `--animate` will store any intermediate (unfinished) latents during the sampling process in CPU memory. After sampling has concluded, an animation (and image grid) will be created in the `/animated` folder in the output directory
- `-cfi`/`--cycle-fresh-image` when combined with image cycles (`-ic`), a new image will be created via text-to-image for each cycle. Can be used to interpolate between prompts purely in text-to-image mode (fixed seed recommended).
### Two-pass generation
- `-spr`/`--second-pass-resize` can be used to perform a second pass on image generation, with image size multiplied by this factor. Enabled for values >1. Initially sampling at a reduced size (e.g. a 'native' image size more common in the model training data) can improve consistency/composition, as well as speed with large resolutions. (Speed can improve when less steps at the larger resolution are ultimately performed)
- `-sps`/`--second-pass-steps` specifies the number of sampling steps used for the second pass (when a second pass is requested via `-spr`).
- `-spc`/`--second-pass-controlnet` switches the second pass to use the first pass image in a selected controlnet (instead of applying image to image). For batch sizes >1, the same (first) image from the first pass will be used as the controlnet input for every batch item (does not apply when running in sequential mode).
### Specifying models
- `-om`/`--online-model` can be used to specify an online model id for acquisition from huggingface hub. This will override the default local (manual) and automatic models. SD1.x only. See: [Specifying a remote model](#specifying-a-remote-model-from-huggingface-sd-1x-only)
- `-lm`/`--local-model` can be used to specify a directory containing local model files. This directory should contain `unet` and `vae` dirs, with a `config.json` and `diffusion_pytorch_model.bin` (or `diffusion_pytorch_model.safetensors`) file each. For SD2.1 or SDXL, additional files will need to be present. See: [Local model installation](#local-model-installation)
### Re-using stored latents
- `-in`/`--interpolate-latents` accepts two image paths for retrieving and interpolating latents from the images. This will only work for images of the same size which have had their latents stored in metadata (`generate.py` does this by default, as it will only increase image size by 50-100kB). While the interpolation occurs in the latent space (after which the VAE is applied to decode individual images), results will usually not differ from crossfading the images in image space directly. Results are saved like in `--animate`.
- `-rnc`/`--re-encode` can be used to specify a path to an image or folder of images, which will be re-encoded using the VAE of the loaded model. This uses the latents stored in image metadata.
### Prompt manipulation
- `-mn`/`--mix-negative-prompts` switches to mixing negative prompts directly into the prompt itself, instead of using them as uncond embeddings. See [Usage & Features](#usage--features)
- `-dnp`/`--default-negative-prompt` can be used to specify a default negative prompt, which will be utilized whenever no negative prompts is given.
- `-cls`/`--clip-layer-skip` can be used to specify a default CLIP (text encoder) skip value if none is specified in the prompt. See [Usage & Features](#usage--features)
- `-sel`/`--static-embedding-length` sets a static prompt embedding length, disabling dynamic length functionality. A value of 77 (text encoder length limit) should be used to reproduce results of previous/other implementations.
- `-mc`/`--mix_concat` switches the prompt mixing mode to concatenate multiple prompt embeddings together instead of calculating a weighted sum. Applied when combining prompts with `;;` or interpolating between prompts. See [Usage & Features](#usage--features)
### LoRA
- `-lop`/`--lora-path` can be used to specify one or more paths to LoRA embedding files (.bin/.pt or .safetensors). The script will attempt to load them via diffusers attn_procs, lora_diffusion or the lora converter.
  - The LoRA converter should be able to convert normal LoRA and convolutional (LoCon) data created through ["kohya-ss/sd-scripts"](https://github.com/kohya-ss/sd-scripts), as well as "Conventional LoRA" (LoRA + LoCon) and (IA)^3 data from [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS).
- `-low`/`--lora-weight` can be used to specify the weights with which LoRA embeddings loaded via `-lop` are applied. Sometimes referred to as 'alpha'.
- `-losc`/`--lora-schedule` will set a schedule for strength of LoRA-like models. Example uses include reducing the changes made to UNET outputs for the last steps with extension models that degrade image quality (e.g.`th7` schedule). This shares scheduler options with `-gsc`, see: [Diffusion Settings](#diffusion-settings). (Effective on adjustment models that contain extra modules, e.g. LoCon. Basic LoRAs are not affected, as their weights are applied directly to the model.)
### ControlNet
- `-com`/`--controlnet-model` can be used to specify a name, path or hub id pointing to a ControlNet model to apply. A directory name in the given repository may be specified via a `|` separator ('<user>/<reponame>|<subdir>'). Custom names can be added to the global variable `CONTROLNET_SHORTNAMES` near the top of `generate.py`. Available names are:
  - SD 1.x:
    - `"canny","depth","hed","mlsd","normal","openpose","scribble","seg"`, which resolve to their respective [lllyasviel/sd-controlnet](https://huggingface.co/lllyasviel) model. For examples, refer to the [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet) repository.
    - `"qrcode"`, which refers to [the SD1.5 qrcode controlnet by monster-labs](https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster)
  - SD 2.1:
    - `"sd21-canny","sd21-depth","sd21-hed","sd21-openpose","sd21-scribble","sd21-zoedepth","sd21-color"`, which resolve to their respective [thibaud/controlnet-sd21](https://huggingface.co/thibaud/controlnet-sd21) model.
  - SDXL:
    - `"xl_canny","xl_canny_mid","xl_canny_small","xl_depth","xl_depth_mid","xl_depth_small","xl_zoedepth"`, which refer to their respective [SDXL controlnets (by diffusers)](https://huggingface.co/collections/diffusers/sdxl-controlnets-64f9c35846f3f06f5abe351f) models. models with suffixes `'_mid'` and `'_small'` are medium-size and small-size distilled variants.
    - `"xl_qrcode"`, an [SDXL qrcode controlnet by monster-labs](https://huggingface.co/monster-labs/control_v1p_sdxl_qrcode_monster)
    - `"xl_inpaint"`, an [SDXL inpainting controlnet by destitech](https://huggingface.co/destitech/controlnet-inpaint-dreamer-sdxl). Note that this model is in early alpha, and requires areas to be inpainted to be blank white in the input.
    - `"xl_softedge"`, a [Softedge controlnet for SDXL by SargeZT](https://huggingface.co/SargeZT/controlnet-sd-xl-1.0-softedge-dexined)
    - `"xl_openpose"`, an [Openpose controlnet for SDXL by thibaud](https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0)
    - `"xl_union","xl_union_promax","xl_openpose_2","xl_tile","xl_depth_2","xl_canny_2","xl_lines","xl_scribble_anime"` which refer to their respective [SDXL controlnets by xinsir](https://huggingface.co/xinsir)
- `-coi`/`--controlnet-input` specifies a path pointing to an input image for the ControlNet.
- `-cop`/`--controlnet-preprocessor` can be used to specify a preprocessor for ControlNet inputs, applied to the ControlNet input image. (To support legacy applications, any preprocessors provided by `"controlnet_aux"` may optionally be specified with a `"detect_"` name prefix.) Available options are:
  - `"canny"`: Applies the Canny edge detector to the image before using it in the ControlNet. Performs a coarse, somewhat textured edge detection. Recommended when feeding regular images into a `"canny"` ControlNet model.
  - `"hed"`: Applies HED edge detection to the input image. Compared to `"canny"`, this will produce soft and smooth edge detection results. Recommended when feeding regular images into an `"hed"` ControlNet model. (Requires `controlnet_aux` library)
  - `"pidi"`: Applies PiDiNet (PixelDifference) edge detection to the image. Recommended for feeding images into `"scribble"` ControlNet models. May also work well with other softedge or lineart models. (Requires `controlnet_aux` library)
  - `"lineart"`: Attempts to produce lineart corresponding to the reference image. Recommended for feeding images into `"lineart"` ControlNet models. (Requires `controlnet_aux` library)
  - `"lineart_anime"`: Attempts to produce lineart corresponding to the reference image, tuned on/for 'anime-style' images. Recommended for feeding images into `"lineart_anime"` ControlNet models. (Requires `controlnet_aux` library)
  - `"mlsd"`: Attempts to produce an M-LSD wireframe segmentation of the input image. Recommended when feeding regular images into an `"mlsd"` ControlNet model. (Requires `controlnet_aux` library)
  - `"midas"`: Computes an estimated depth map via MiDaS. Recommended for feeding images into `"midas-depth"` or otherwise unspecified `"depth"` ControlNet models. (Requires `controlnet_aux` library)
  - `"zoe"`: Computes an estimated depth map via ZoeDepth. This is usually more precise and detailed than MiDaS depth estimates. Recommended for feeding images into `"zoe"` or `"zoedepth"` ControlNet models. As most generic `"depth"` ControlNet models tend to be trained on MiDaS depth estimates, using them with ZoeDepth estimation may yield degraded performance. (Requires `controlnet_aux` library)
  - `"leres"`: Computes an estimated depth map via LeReS 3d scene shape estimation. This may provide more details than other depth estimation methods, especially on depth gradients. (Requires `controlnet_aux` library)
  - `"normalbae"`: Computes an estimated normal map via 'Bae's estimation method'. Recommended for feeding images into `"normal"` or `"normalbae"` ControlNet models. (Requires `controlnet_aux` library)
  - `"pose"`: Attempts to extract an openpose bone image from the input image. Can be used to perform 'pose-transfer'. Recommended when feeding regular images into an `"openpose"` ControlNet model. (Requires `controlnet_aux` library)
  - `"shuffle"`: 'Shuffles' image content by 'distorting' it via a random flow pattern. Recommended for feeding images into `"shuffle"` ControlNet models. (Requires `controlnet_aux` library)
  - `"blur"`: Blurs image content using gaussian blur. Recommended for use with 'blur' or 'tile deblur' ControlNet models.
- `-cost`/`--controlnet-strength` specifies the strength (guidance scale/cond_scale) with which the ControlNet guidance is applied.
- `-cosc`/`--controlnet-schedule` can be used to specify a schedule for variable ControlNet strength. Default (None) corresponds to no schedule. This shares scheduler options with `-gsc`, see: [Diffusion Settings](#diffusion-settings).

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
  - `FLAG_POTENTIAL_NSFW` can be set to `False` to entirely disable the content check. THIS IS GENERALLY NEVER RECOMMENDED. Instead, a safety level of at least 1 should be set depending on usage requirements.
  - `SAFETY_PROCESSING_LEVEL_NSFW_CHANNEL` sets the safety processing level applied to outputs in NSFW channels. See `-spl` in [Additional Flags](#additional-flags) for details.
  - `SAFETY_PROCESSING_LEVEL_SFW_CHANNEL` is only relevant if usage in SFW channels is manually enabled, seting the safety processing level applied to outputs in SFW channels. See `-spl` in [Additional Flags](#additional-flags) for details.
  - `PERMIT_SFW_CHANNEL_USAGE` can be set to `True` to permit usage in SFW channels. ENABLE AT YOUR OWN RISK! This will require additional moderation and content screening. False negatives can occur in the safety checker, which can cause potentially NSFW content to be sent in SFW channels.
  - `USE_HALF_LATENTS` see: `--latents-half` in [Device, Performace and Optimization settings](#device-performace-and-optimization-settings)
  - `ATTENTION_SLICING` see: `-as` in [Device, Performace and Optimization settings](#device-performace-and-optimization-settings)
  - `CPU_OFFLOAD` see: `-co` in [Device, Performace and Optimization settings](#device-performace-and-optimization-settings)
  - `PERMIT_RELOAD` can be set to `True` to allow users to switch the current model, toggle CPU offloading and set attention slicing via the `/reload` command.
    - If set to `True`, `permittel_local_model_paths` specifies a whitelist of local model names with their respective model paths (see: [Local model installation](#local-model-installation)), while `permitted_model_ids` specifies a whitelist of names with respective huggingface hub model ids (see: [Specifying a remote model](#specifying-a-remote-model-from-huggingface-sd-1x-only))
  - `PERMITTED_LORAS` specifies a whitelist of LoRA embeddings which can optionally be loaded when reloading. By default, this will be any `lora/*.safetensors` file.
  - `INPUT_IMAGES_BACKGROUND_COLOR` can be used to set a the frame highlighting color of input images (used to visually separate inputs from outputs)
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
  - `/advanced <text prompt> <width> <height> <seed> <guidance_scale> <steps> <img2img_strength> <Image attachment> <amount> <scheduler> <gs_schedule> <static_length> <controlnet> <controlnet_sd2> <controlnet_input> <controlnet_strength> <controlnet_schedule> <second_pass_resize> <second_pass_steps> <second_pass_ctrl> <use_karras_sigmas> <lora_schedule>`
    - `Width` and `height` are specified either as pixels (for values >64), or as a multiplier of 64, offset from 512x512. A `width` of `3` and `height` of `-2` will result in an image which is `512+64*3 = 704` pixels wide and `512-64*2 = 384` pixels high
    - If seeds are set to a value below `0`, the seed is randomized. The randomly picked seed will be returned in the image response.
    - `scheduler`, `gs_schedule`, `controlnet` and `controlnet_schedule` display available options.
    - Unless a source image is attached, `img2img_strength` is ignored.
    - Steps are limited to `150` by default.
    - By default, available ControlNets will include the default options available for `-com` ([ControlNet](#controlnet)), as well as variants with the prefix `process_`, which apply the respective preprocessor (`-cop`, [ControlNet](#controlnet)) of the ControlNet model.
      - Controlnet options have been split into separate arguments for SD1.x and SD2.x, as keeping them merged would exceed the 25 option limit of a command parameter set by discord.
  - `/reload <model name> <enable cpu offload> <attention slicing> <default CLIP skip> <lora1> <lora2> <lora3> <lora1_weight> <lora2_weight> <lora3_weight>` if `PERMIT_RELOAD` is changed to True, this can be used to (re-)load the model from a selection of available models (see above).
    -  Up to three of any available LoRA models can be loaded with a respective weight/alpha. If being able to load more than three LoRA embeddings is required, the command parameters can easily be extended.
  - `/default_negative <negative prompt>` can be used to set a default negative prompt (see: `-dnp`, [Additional flags](#additional-flags)). If <negative_prompt> is not specified, the default negative prompt will be reset.
- All commands come with a short documentation of their available parameters.

#
# Notes
- For more information about [Stable Diffusion](https://github.com/CompVis/stable-diffusion) and the [huggingface diffusers model of Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion-v1-4), including the license, limitations, and capabilities of the systems utilized, check out the respective links.
- Text to image and image to image implementations are derived from the pipeline implementations of the [huggingface diffusers library](https://github.com/huggingface/diffusers)
