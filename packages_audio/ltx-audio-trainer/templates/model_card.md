---
tags:
  - ltx-2
  - ltx-audio
  - text-to-audio
  - audio-generation
pinned: true
language:
  - en
license: other
pipeline_tag: text-to-audio
library_name: diffusers
---

# {model_name}

This is a fine-tuned version of [`{base_model}`]({base_model_link}) trained on custom data.

## Model Details

- **Base Model:** [`{base_model}`]({base_model_link})
- **Training Type:** {training_type}
- **Training Steps:** {training_steps}
- **Learning Rate:** {learning_rate}
- **Batch Size:** {batch_size}

## Sample Outputs

{sample_entries}

## Usage

This model is designed to be used with an LTX-2 audio generation pipeline.

### 🔌 Using Trained LoRAs in ComfyUI

In order to use the trained LoRA in ComfyUI, follow these steps:

1. Copy your trained LoRA checkpoint (`.safetensors` file) to the `models/loras` folder in your ComfyUI installation.
2. In your ComfyUI workflow:
    - Add the "Load LoRA" node to choose your LoRA file
    - Connect it to the "Load Checkpoint" node to apply the LoRA to the base model

You can find reference text-to-audio workflows in the
official [LTX-2 repository](https://github.com/Lightricks/LTX-2).

### Example Prompts

{validation_prompts}


This model inherits the license of the base model ([`{base_model}`]({base_model_link})).

## Acknowledgments

- Base model: [Lightricks](https://huggingface.co/Lightricks/LTX-2)
- Trainer: [LTX-2](https://github.com/Lightricks/LTX-2)
