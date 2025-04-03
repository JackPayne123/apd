# %%
import torch
from simple_stories_train.models.llama import Llama
from simple_stories_train.models.model_configs import MODEL_CONFIGS
from transformers import AutoTokenizer

from spd.experiments.lm.models import SSModel, create_gate_proj_components

# %%
# Select the model size you want to use
model_size = "1.25M"  # Options: "35M", "30M", "11M", "5M", "1.25M"

# Load model configuration
model_config = MODEL_CONFIGS[model_size]

# Load appropriate model
model_path = f"chandan-sreedhara/SimpleStories-{model_size}"
model = Llama.from_pretrained(model_path, model_config)
# model.to("cuda")
model.eval()
# %%

ss_model = SSModel(model)

m = 17
# Create components with rank=10 (adjust as needed)
gate_proj_components = create_gate_proj_components(model, rank=m)

# %%
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)

# Define your prompt
prompt = "The curious cat looked at the"

# IMPORTANT: Use tokenizer without special tokens
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
# input_ids = inputs.input_ids.to("cuda")
input_ids = inputs.input_ids
# Targets should be the inputs shifted by one (we will later ignore the last input token)
targets = input_ids[:, 1:]
input_ids = input_ids[:, :-1]

# IMPORTANT: Set correct EOS token ID (not the default from tokenizer)
eos_token_id = 1

# %%

# logits, _ = ss_model.forward(input_ids, components=gate_proj_components)
logits, _ = ss_model.forward(input_ids)
print("inputs_shape", input_ids.shape)
print("logits", logits)
print("logits shape", logits.shape)

logits, _ = ss_model.forward_with_components(input_ids, components=gate_proj_components)

print("Component logits shape", logits.shape)
print("Component logits", logits)

# Create some dummy masks
masks = {
    f"model.transformer.h.{i}.mlp.gate_proj": torch.randn(1, input_ids.shape[-1], m)
    for i in range(len(model.transformer.h))
}

logits, _ = ss_model.forward_with_components(
    input_ids, components=gate_proj_components, masks=masks
)

print("Masked component logits shape", logits.shape)
print("Masked component logits", logits)
# %%
