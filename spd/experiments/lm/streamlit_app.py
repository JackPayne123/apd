"""
To run this app, run the following command:

```bash
    streamlit run spd/experiments/lm/streamlit_app.py -- --model_path "wandb:spd-lm/runs/151bsctx"
```
"""

import argparse
from collections.abc import Iterator
from typing import Any

import streamlit as st
import streamlit_antd_components as sac
import torch
from jaxtyping import Float, Int
from simple_stories_train.dataloaders import DatasetConfig, create_data_loader
from torch import Tensor
from transformers import AutoTokenizer

from spd.configs import LMTaskConfig
from spd.experiments.lm.models import LinearComponentWithBias, SSModel
from spd.log import logger
from spd.models.components import Gate, GateMLP
from spd.run_spd import calc_component_acts, calc_masks
from spd.types import ModelPath

DEFAULT_MODEL_PATH: ModelPath = "wandb:spd-lm/runs/151bsctx"


# --- Initialization and Data Loading ---
@st.cache_resource(show_spinner="Loading model and data...")
def initialize(model_path: ModelPath) -> dict[str, Any]:
    """
    Loads the model, tokenizer, config, and evaluation dataloader.
    Cached by Streamlit based on the model_path.
    """
    device = "cpu"  # Use CPU for the Streamlit app
    logger.info(f"Initializing app with model: {model_path} on device: {device}")
    ss_model, config, _ = SSModel.from_pretrained(model_path)
    ss_model.to(device)
    ss_model.eval()

    assert isinstance(config.task_config, LMTaskConfig), (
        "Task config must be LMTaskConfig for this app."
    )

    # Derive tokenizer path (adjust if stored differently)
    tokenizer_path = f"chandan-sreedhara/SimpleStories-{config.task_config.model_size}"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, legacy=False)

    # Create eval dataloader config
    eval_data_config = DatasetConfig(
        name=config.task_config.dataset_name,
        tokenizer_file_path=None,
        hf_tokenizer_path=tokenizer_path,
        split=config.task_config.eval_data_split,
        n_ctx=config.task_config.max_seq_len,
        is_tokenized=False,
        streaming=False,  # Non-streaming might be simpler for iterator reset
        column_name="story",
    )

    # Create the dataloader iterator
    def create_dataloader_iter() -> Iterator[dict[str, Int[Tensor, "1 seq_len"]]]:
        logger.info("Creating new dataloader iterator.")
        dataloader, _ = create_data_loader(
            dataset_config=eval_data_config,
            batch_size=1,  # Always use batch size 1 for this app
            buffer_size=config.task_config.buffer_size,
            global_seed=config.seed,  # Use same seed for reproducibility
            ddp_rank=0,
            ddp_world_size=1,
        )
        return iter(dataloader)

    # Extract components and gates
    gates: dict[str, Gate | GateMLP] = {
        k.removeprefix("gates.").replace("-", "."): v for k, v in ss_model.gates.items()
    }
    components: dict[str, LinearComponentWithBias] = {
        k.removeprefix("components.").replace("-", "."): v for k, v in ss_model.components.items()
    }
    target_layer_names = sorted(list(components.keys()))

    logger.info(f"Initialization complete for {model_path}.")
    return {
        "model": ss_model,
        "tokenizer": tokenizer,
        "config": config,
        "dataloader_iter_fn": create_dataloader_iter,  # Store the function to create iter
        "gates": gates,
        "components": components,
        "target_layer_names": target_layer_names,
        "device": device,
    }


def load_next_prompt() -> None:
    """Loads the next prompt, calculates masks, and prepares token data."""
    logger.info("Loading next prompt.")
    app_data = st.session_state.app_data
    dataloader_iter = st.session_state.dataloader_iter  # Get current iterator

    try:
        batch = next(dataloader_iter)
        input_ids: Int[Tensor, "1 seq_len"] = batch["input_ids"].to(app_data["device"])
    except StopIteration:
        logger.warning("Dataloader iterator exhausted. Throwing error.")
        st.error("Failed to get data even after resetting dataloader.")
        return

    st.session_state.current_input_ids = input_ids

    # Decode tokenized IDs to get the transformed text
    st.session_state.transformed_prompt_text = app_data["tokenizer"].decode(
        input_ids[0], skip_special_tokens=True
    )

    # Calculate activations and masks
    with torch.no_grad():
        (_, _), pre_weight_acts = app_data["model"].forward_with_pre_forward_cache_hooks(
            input_ids, module_names=list(app_data["components"].keys())
        )
        As = {
            module_name: v.linear_component.A for module_name, v in app_data["components"].items()
        }
        target_component_acts = calc_component_acts(pre_weight_acts=pre_weight_acts, As=As)
        masks, _ = calc_masks(
            gates=app_data["gates"],
            target_component_acts=target_component_acts,
            attributions=None,
            detach_inputs=True,  # No gradients needed
        )
    st.session_state.current_masks = masks  # Dict[str, Float[Tensor, "1 seq_len m"]]

    # Prepare token data for display
    token_data = []
    tokenizer = app_data["tokenizer"]
    for i, token_id in enumerate(input_ids[0]):
        # Decode individual token - might differ slightly from full decode for spaces etc.
        decoded_token_str = tokenizer.decode([token_id])
        token_data.append({"id": token_id.item(), "text": decoded_token_str, "index": i})
    st.session_state.token_data = token_data

    # Reset selections
    st.session_state.selected_token_index = None
    st.session_state.selected_layer_name = None
    logger.info("Finished loading next prompt and calculating masks.")


def set_selected_token(index: int) -> None:
    """Callback function to set the selected token index."""
    # Check if the index is valid before setting
    if 0 <= index < len(st.session_state.token_data):
        logger.debug(f"Token {index} selected.")
        st.session_state.selected_token_index = index
    else:
        logger.debug(f"Invalid index ({index}) received or no token clicked.")


# --- Main App UI ---
def run_app(args: argparse.Namespace) -> None:
    """Sets up and runs the Streamlit application."""
    st.set_page_config(layout="wide")
    st.title("LM Component Activation Explorer")

    # Initialize model, data, etc. (cached)
    st.session_state.app_data = initialize(args.model_path)
    app_data = st.session_state.app_data
    st.caption(f"Model: {args.model_path}")

    # Initialize session state variables if they don't exist
    if "transformed_prompt_text" not in st.session_state:
        st.session_state.transformed_prompt_text = None
    if "token_data" not in st.session_state:
        st.session_state.token_data = None
    if "current_masks" not in st.session_state:
        st.session_state.current_masks = None
    if "selected_token_index" not in st.session_state:
        st.session_state.selected_token_index = None
    if "selected_layer_name" not in st.session_state:
        if app_data["target_layer_names"]:
            st.session_state.selected_layer_name = app_data["target_layer_names"][0]
        else:
            st.session_state.selected_layer_name = None
    # Initialize the dataloader iterator in session state
    if "dataloader_iter" not in st.session_state:
        st.session_state.dataloader_iter = app_data["dataloader_iter_fn"]()

    # --- Prompt Area ---
    st.button("Load Initial / Next Prompt", on_click=load_next_prompt)

    # Display Transformed (Decoded) Prompt using Clickable Tokens
    if st.session_state.token_data:
        st.subheader("Prompt (Encoded->Decoded, Click Tokens Below)")

        # Use sac.buttons to create clickable text segments
        clicked_token_index = sac.buttons(
            items=[
                sac.ButtonsItem(label=token_info["text"])
                for i, token_info in enumerate(st.session_state.token_data)
            ],
            index=st.session_state.selected_token_index,
            format_func=None,
            align="left",
            variant="text",
            size="xs",
            gap=1,
            use_container_width=False,
            return_index=True,
            key="token_buttons",
            radius=1,
        )

        # Update selected token based on click
        if clicked_token_index != st.session_state.selected_token_index:
            set_selected_token(clicked_token_index)
            st.rerun()

    st.divider()

    # --- Token Information Area ---
    if st.session_state.selected_token_index is not None:
        idx = st.session_state.selected_token_index
        # Ensure token_data is loaded before accessing
        if st.session_state.token_data and idx < len(st.session_state.token_data):
            token_info = st.session_state.token_data[idx]
            token_text = token_info["text"]

            st.header(f"Token Info: '{token_text}' (Position: {idx}, ID: {token_info['id']})")

            # Layer Selection Dropdown
            st.selectbox(
                "Select Layer to Inspect:",
                options=app_data["target_layer_names"],
                key="selected_layer_name",  # Binds selection to session state
            )

            # Display Layer-Specific Info if a layer is selected
            if st.session_state.selected_layer_name:
                layer_name = st.session_state.selected_layer_name
                logger.debug(f"Displaying info for token {idx}, layer {layer_name}")

                if st.session_state.current_masks is None:
                    st.warning("Masks not calculated yet. Please load a prompt.")
                    return

                layer_mask_tensor: Float[Tensor, "1 seq_len m"] = st.session_state.current_masks[
                    layer_name
                ]
                token_mask: Float[Tensor, " m"] = layer_mask_tensor[0, idx, :]

                # Find active components (mask > 0)
                active_indices_layer: Int[Tensor, " n_active"] = torch.where(token_mask > 0)[0]
                n_active_layer = len(active_indices_layer)

                st.metric(f"Active Components in {layer_name}", n_active_layer)

                st.subheader("Active Component Indices")
                if n_active_layer > 0:
                    # Convert to NumPy array and reshape to a column vector (N x 1)
                    active_indices_np = active_indices_layer.cpu().numpy().reshape(-1, 1)
                    # Pass the NumPy array directly and configure the column header
                    st.dataframe(
                        active_indices_np,
                        height=300,
                        use_container_width=False,
                        column_config={0: "Component Index"},  # Rename the first column (index 0)
                    )
                else:
                    st.write("No active components for this token in this layer.")

                # Extensibility Placeholder
                st.subheader("Additional Layer/Token Analysis")
                st.write(
                    "Future figures and analyses for this specific layer and token will appear here."
                )
        else:
            # Handle case where selected_token_index might be invalid after data reload
            st.warning("Selected token index is out of bounds. Please select a token again.")
            st.session_state.selected_token_index = None  # Reset selection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Streamlit app to explore LM component activations."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path or W&B reference to the trained SSModel. Default: {DEFAULT_MODEL_PATH}",
    )
    args = parser.parse_args()

    run_app(args)
