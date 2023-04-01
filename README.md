# Stable Diffusion Tests
I became interested in using SD to generate images for military applications. Most of the resources are taken from 4chan's NSFW boards, as anons use SD to make hentai. Anyhow, the techniques from these weirdos are applicable to a variety of applications, most specifically LORAs, which are like model fine-tuners... idea is to work with specific LORAs (e.g., military vehicles, aircraft, weapons, etc.) to generate synthetic image data for training vision models. Training new, useful LORAs is also of interest. Later stuff may inlcude inpainting for perturbation.

-TP

# Play With It!
What can you actually do with SD? Huggingface and some others have some apps in-browser for you. Play around with them to see the power! What we will do in this guide is get the full, extensible WebUI to allow us to do anything we want.
* [Huggingface Text to Image SD Playground](https://huggingface.co/spaces/stabilityai/stable-diffusion)
* [Dreamstudio Text to Image SD App](https://beta.dreamstudio.ai/generate)
* [Dezgo Text to Image SD App](https://dezgo.com/)
* [Huggingface Image to Image SD Playground](https://huggingface.co/spaces/huggingface-projects/diffuse-the-rest)
* [Huggingface Inpainting Playground](https://huggingface.co/spaces/fffiloni/stable-diffusion-inpainting)

# Steps I Took
It's somewhat daunting to get into this... but 4channers have done a good job making this approachable. Below are the steps I took, in the simplest terms. Your intent is to get the Stable Diffusion WebUI (built with Gradio) running locally so you can start prompting and making images.

## Setting up Local GPU Usage (~1 hour)
We will do Google Colab Pro setup later, so we can run SD on any device anywhere we want; but to start, let's get the it setup on a PC. You need 16GB RAM, a GPU with 2GB VRAM, Linux or Windows 7+ and 20+GB disk space.
1. Finish the [starting setup guide](https://rentry.org/voldy)
    * I followed this up to step 7, after which it goes into the hentai stuff
    * Step 3 takes 15-45 minutes on average Internet speed, as the models are 5+ GB each
    * Step 7 can take upwards of half an hour and may seem "stuck" in the CLI
    * In step 3 I downloaded SD1.5, not the 2.x versions, as 1.5 produces much better results
    * In step 3, I also downloaded the "Deliberate" and "Dreamshaper" models, both of which have a focus on realism, with the latter, scenery and fantasy
    * [Civitai](https://civitai.com/) has all the SD models; it's like HuggingFace but for SD specifically
2. Verify the WebUI works
    1. Copy the URL the CLI outputs once done, e.g., ```127.0.0.1:7860``` (do **NOT** use Ctrl + C because this command can close the CLI)
    2. Paste into browser and voila; try a prompt and you're off to the races

![](1.png)

3. Read up on prompting techniques, because there are lots of things to know (e.g., positive prompt vs. negative prompt, sampling steps, sampling method, etc.)
    * [Definitive SD Prompting Guide](https://stable-diffusion-art.com/prompt-guide/)
    * [4chan prompting tips](https://rentry.org/hdgpromptassist#terms) (some images lower on the page are NSFW)
4. Read up on SD knowledge in general:
    * [Stable Diffusion Compendium](https://www.sdcompendium.com/doku.php?id=start)
    * [Stable Diffusion Links Hub](https://rentry.org/rentrysd)
