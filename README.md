# Run DreamO with Cog and Replicate

[![Replicate](https://replicate.com/zsxkib/dream-o/badge)](https://replicate.com/zsxkib/dream-o)

DreamO is an image model from ByteDance that you can use with Cog. It's good for creating images from text descriptions, using one image to set the style for another, or making sure a face stays the same across different images.

This guide shows you how to get DreamO running with Cog and how to put your model on Replicate.

## What you need

First, you'll need to install Cog. You can find instructions in the [official Cog installation guide](https://github.com/replicate/cog#install). You'll usually need Docker installed too.

## How to run the model

You can run the DreamO model with the `cog predict` command. Here's what the different settings do:

*   `prompt` (text): The main text you want the model to create an image from.
*   `negative_prompt` (text, optional): Words to tell the model what *not* to include in the image. It has a default list of terms like "unrealistic, fake, low quality".
*   `ref_image` (file, optional): An image file you want to use as a reference.
*   `ref_task` (text): Tells the model how to use the `ref_image`. Your options are:
    *   `id`: Keeps the face from your `ref_image`. This is the default.
    *   `ip`: Uses your `ref_image` to make a new image, and tries to remove the background from the reference.
    *   `style`: Uses your `ref_image` just for its style.
*   `task` (text): Sets the main job for the model:
    *   `text_guided`: Mainly uses your `prompt` to create the image. This is the default.
    *   `image_guided`: Mainly uses your `ref_image` and the `ref_task` setting.
    *   `mix`: Balances text and image guidance.
*   `style_strength` (number): How much of the `ref_image`'s style to apply. Use a number between 0.0 and 1.0. Default is 0.5.
*   `ip_scale` (number): How much to scale the reference image if `ref_task` is `ip` or `style`. Use a number between 0.0 and 1.0. Default is 0.6.
*   `width` (number): How wide you want the final image to be, in pixels. Default is 1024.
*   `height` (number): How tall you want the final image to be, in pixels. Default is 1024.
*   `num_inference_steps` (number): Number of steps the model takes to create the image. More steps can mean better quality but take longer. Default is 30.
*   `guidance_scale` (number): How much the model should stick to your prompt. Higher values mean stricter adherence. Default is 7.5.
*   `seed` (number, optional): A number to make your image generation repeatable. If you don't set one, it will pick a random seed.
*   `output_format` (text): The file type for the output image. Choices are `webp`, `jpg`, or `png`. Default is `webp`.
*   `output_quality` (number): For `webp` or `jpg` images, this sets the compression quality from 1 to 100. Default is 80.

**Examples**

Here are a few ways you can use `cog predict`. These examples use image URLs, but you can also use files from your computer.

**Keep a face consistent**

This example creates an image of "A beautiful woman..." and tries to use the face from `woman1.png`.

```bash
cog predict -i prompt="A beautiful woman, masterpiece, stunning, ultra high res, cinematic lighting" \
    -i ref_image="https://raw.githubusercontent.com/bytedance/DreamO/main/example_inputs/woman1.png" \
    -i ref_task="id" \
    -i task="text_guided" \
    -i seed=0
```

**Use an image for composition**

This example uses `dog1.png` as a reference, tries to remove its background, and then creates an image based on the prompt "A Corgi puppy...".

```bash
cog predict -i prompt="A Corgi puppy, masterpiece, stunning, ultra high res, cinematic lighting" \
    -i ref_image="https://raw.githubusercontent.com/bytedance/DreamO/main/example_inputs/dog1.png" \
    -i ref_task="ip" \
    -i task="text_guided" \
    -i ip_scale=0.7 \
    -i seed=1
```

**Transfer a style**

This example creates an image of "A sports car..." using the style from `style1.png`.

```bash
cog predict -i prompt="A sports car, masterpiece, stunning, ultra high res, cinematic lighting" \
    -i ref_image="https://raw.githubusercontent.com/bytedance/DreamO/main/example_inputs/style1.png" \
    -i ref_task="style" \
    -i task="text_guided" \
    -i style_strength=0.6 \
    -i seed=2
```

**Mix text and image guidance**

This example uses mixed guidance with a reference image to keep the face consistent for "A beautiful man in Van Gogh style...".

```bash
cog predict -i prompt="A beautiful man in Van Gogh style, masterpiece, stunning, ultra high res, cinematic lighting" \
    -i ref_image="https://raw.githubusercontent.com/bytedance/DreamO/main/example_inputs/man1.png" \
    -i ref_task="id" \
    -i task="mix" \
    -i seed=3
```

## Put your model on Replicate

Once your model works the way you want with Cog, you can share it on Replicate.

1.  **Log in to Replicate:**
    ```bash
    cog login
    ```
2.  **Push your model:**
    Replace `your-username/your-model-name` with your Replicate username and the name you want for your model.
    ```bash
    cog push r8.im/your-username/your-model-name
    ```
    For example, if your Replicate username is `zsxkib` and you call your model `dream-o`, you would run:
    ```bash
    cog push r8.im/zsxkib/dream-o
    ```

## License

MIT

---

⭐ Star this on [GitHub](https://github.com/zsxkib/cog-DreamO)!

👋 Follow `zsxkib` on [Twitter/X](https://twitter.com/zsakib_)