import argparse
import logging
from collections.abc import Iterator
from typing import Any, cast

import gradio as gr
import torch
from jaxtyping import Float, Int
from simple_stories_train.dataloaders import DatasetConfig, create_data_loader
from torch import Tensor
from transformers import AutoTokenizer

from spd.configs import Config, LMTaskConfig
from spd.experiments.lm.models import LinearComponentWithBias, SSModel
from spd.models.components import Gate, GateMLP
from spd.run_spd import calc_component_acts, calc_masks
from spd.types import ModelPath

# --- Configuration & Constants ---

DEFAULT_MODEL_PATH: ModelPath = "wandb:spd-lm/runs/151bsctx"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Data Structures ---

# Structure to hold information mapping character spans to tokens
TokenMapItem = dict[str, Any]  # Keys: 'text', 'span': tuple[int, int], 'index': int, 'id': int
TokenMap = list[TokenMapItem]

# Structure for Gradio state
AppState = dict[str, Any]  # Keys: 'model', 'tokenizer', 'config', 'gates', 'components', etc.


# --- Core Functions ---


@torch.no_grad()
def load_resources(model_path: ModelPath, device: str) -> AppState:
    """Loads the model, tokenizer, config, components, and gates."""
    logger.info(f"Loading resources for model: {model_path} on device: {device}")
    ss_model, config, _ = SSModel.from_pretrained(model_path)
    ss_model.to(device)
    ss_model.eval()

    assert isinstance(config.task_config, LMTaskConfig), (
        "Task config must be LMTaskConfig for this app."
    )

    # Derive tokenizer path
    tokenizer_path = f"chandan-sreedhara/SimpleStories-{config.task_config.model_size}"
    # Use the base tokenizer from AutoTokenizer for consistency if needed,
    # but create_data_loader might load its own. Ensure they are compatible.
    # For decoding/mapping, AutoTokenizer is convenient.
    hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, legacy=False)

    # Extract components and gates
    gates: dict[str, Gate | GateMLP] = {
        k.removeprefix("gates.").replace("-", "."): v for k, v in ss_model.gates.items()
    }
    components: dict[str, LinearComponentWithBias] = {
        k.removeprefix("components.").replace("-", "."): v for k, v in ss_model.components.items()
    }
    target_layer_names = sorted(list(components.keys()))

    logger.info(f"Finished loading resources for {model_path}.")
    return {
        "model": ss_model,
        "tokenizer": hf_tokenizer,  # Use HF Tokenizer for decoding/mapping
        "config": config,
        "gates": gates,
        "components": components,
        "target_layer_names": target_layer_names,
        "device": device,
        "tokenizer_path": tokenizer_path,  # Store path for dataloader
    }


def create_eval_dataloader_iter(
    app_state: AppState,
) -> Iterator[dict[str, Int[Tensor, "1 seq_len"]]]:
    """Creates a new iterator for the evaluation dataloader."""
    config: Config = app_state["config"]
    task_config: LMTaskConfig = cast(LMTaskConfig, config.task_config)
    tokenizer_path: str = app_state["tokenizer_path"]
    logger.info("Creating new evaluation dataloader iterator.")

    eval_data_config = DatasetConfig(
        name=task_config.dataset_name,
        tokenizer_file_path=None,  # Use HF tokenizer path
        hf_tokenizer_path=tokenizer_path,
        split=task_config.eval_data_split,
        n_ctx=task_config.max_seq_len,
        is_tokenized=False,  # Tokenize on the fly
        streaming=True,  # Use streaming as requested
        column_name="story",
        seed=config.seed,  # Use same seed for reproducibility if needed
    )

    dataloader, _ = create_data_loader(
        dataset_config=eval_data_config,
        batch_size=1,  # Always use batch size 1 for this app
        buffer_size=task_config.buffer_size,
        global_seed=config.seed,
        ddp_rank=0,
        ddp_world_size=1,
    )
    # Make the dataloader an explicit iterator
    return iter(dataloader)


def get_token_mapping(
    tokenizer: AutoTokenizer, input_ids: Int[Tensor, "1 seq_len"]
) -> tuple[str, TokenMap]:
    """
    Decodes input_ids and creates a mapping from character spans to token info.
    Handles potential decoding artifacts like extra spaces.
    """
    ids_list = input_ids[0].tolist()
    full_text = tokenizer.decode(ids_list, skip_special_tokens=True)
    logger.debug(f"Full decoded text length: {len(full_text)}")

    token_map: TokenMap = []
    current_char_index = 0

    for token_idx, token_id in enumerate(ids_list):
        # Decode individual token *without* special tokens or added spaces
        # Note: This might differ slightly from full decode for some tokenizers (e.g., SentencePiece)
        # We prioritize matching the token's contribution to the full decoded string.
        token_text = tokenizer.decode(
            [token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Find the *next* occurrence of this token's text in the full string
        try:
            # Find the start index, searching from the current position
            start_char = full_text.index(token_text, current_char_index)
            end_char = start_char + len(token_text)

            # Store mapping info
            token_map_item: TokenMapItem = {
                "text": token_text,
                "span": (start_char, end_char),
                "index": token_idx,
                "id": token_id,
            }
            token_map.append(token_map_item)
            # logger.debug(f"Mapped token {token_idx} (ID: {token_id}, Text: '{token_text}') to span {token_map_item['span']}")

            # Update current character index for the next search
            current_char_index = end_char

        except ValueError:
            # This can happen if the individual token decode differs significantly
            # from its representation in the full decode (e.g., spaces, merges)
            logger.warning(
                f"Could not find token_text='{token_text}' (ID: {token_id}, Index: {token_idx}) "
                f"in remaining full_text='{full_text[current_char_index:]}'. Skipping token mapping."
            )
            # Attempt to gracefully handle by skipping or trying alternative decodes if necessary
            # For now, we just log and potentially skip. A robust solution might require
            # tokenizer-specific logic or offset mapping if available.

    # Verification step (optional but recommended)
    if current_char_index != len(full_text) and len(token_map) == len(ids_list):
        logger.warning(
            f"Final character index {current_char_index} does not match full text length {len(full_text)}. Mapping might be imperfect."
        )
    elif len(token_map) != len(ids_list):
        logger.warning(
            f"Mapped {len(token_map)} tokens, but expected {len(ids_list)}. Mapping is incomplete."
        )

    return full_text, token_map


@torch.no_grad()
def calculate_masks_for_batch(
    app_state: AppState, input_ids: Int[Tensor, "1 seq_len"]
) -> dict[str, Float[Tensor, "1 seq_len m"]]:
    """Performs forward pass and calculates masks for the given input_ids."""
    model: SSModel = app_state["model"]
    components: dict[str, LinearComponentWithBias] = app_state["components"]
    gates: dict[str, Gate | GateMLP] = app_state["gates"]
    device: str = app_state["device"]

    input_ids = input_ids.to(device)

    logger.info("Running forward pass to get activations...")
    (_, _), pre_weight_acts = model.forward_with_pre_forward_cache_hooks(
        input_ids, module_names=list(components.keys())
    )
    logger.info("Calculating component activations...")
    As = {module_name: v.linear_component.A for module_name, v in components.items()}
    target_component_acts = calc_component_acts(pre_weight_acts=pre_weight_acts, As=As)

    logger.info("Calculating masks...")
    masks, _ = calc_masks(
        gates=gates,
        target_component_acts=target_component_acts,
        attributions=None,
        detach_inputs=True,  # No gradients needed for inference
    )
    logger.info("Mask calculation complete.")
    # Ensure masks are on CPU for Gradio state if needed, although calculations were on device
    return {k: v.cpu() for k, v in masks.items()}


def update_token_display(
    selected_token_info: TokenMapItem | None,
    selected_layer_name: str | None,
    current_masks: dict[str, Float[Tensor, "1 seq_len m"]] | None,
) -> tuple[str, str, Any, bool]:
    """
    Generates display updates for the token info area based on selection.
    Returns: (token_summary_md, active_comp_summary_md, active_indices_df_data, area_visible)
    """
    if not selected_token_info or not selected_layer_name or not current_masks:
        logger.debug("Update condition not met, hiding token info area.")
        return "", "", None, False  # Hide area if no token/layer/masks

    token_idx = selected_token_info["index"]
    token_id = selected_token_info["id"]
    token_text = selected_token_info["text"]

    logger.info(
        f"Updating display for token {token_idx} ('{token_text}'), layer {selected_layer_name}"
    )

    token_summary_md = f"**Token Info:** '{token_text}' (Position: {token_idx}, ID: {token_id})"

    if selected_layer_name not in current_masks:
        logger.warning(f"Selected layer {selected_layer_name} not found in current masks.")
        return token_summary_md, "Error: Layer not found in masks.", None, True

    layer_mask_tensor: Float[Tensor, "1 seq_len m"] = current_masks[selected_layer_name]
    # Ensure tensor is on CPU before indexing if it wasn't already
    token_mask: Float[Tensor, " m"] = layer_mask_tensor[0, token_idx, :].cpu()

    # Find active components (mask > 0)
    active_indices_layer: Int[Tensor, " n_active"] = torch.where(token_mask > 0)[0]
    n_active_layer = len(active_indices_layer)

    active_comp_summary_md = f"**Active Components in {selected_layer_name}:** {n_active_layer}"

    active_indices_df_data = None
    if n_active_layer > 0:
        # Convert to NumPy array and reshape for DataFrame (N x 1)
        active_indices_np = active_indices_layer.cpu().numpy().reshape(-1, 1)
        active_indices_df_data = active_indices_np
        logger.debug(f"Found {n_active_layer} active components.")
    else:
        logger.debug("No active components found.")

    return token_summary_md, active_comp_summary_md, active_indices_df_data, True


# --- Gradio Application ---

# Store the iterator globally (or in a mutable container) for access within handlers
# This is suitable for single-session Gradio apps.
global_iterator_store: dict[str, Iterator[dict[str, Int[Tensor, "1 seq_len"]]] | None] = {
    "iterator": None
}


def build_gradio_app(args: argparse.Namespace) -> gr.Blocks:
    """Builds the Gradio Blocks interface."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load resources once
    initial_app_state = load_resources(args.model_path, device)
    # Create the initial iterator and store it globally
    global_iterator_store["iterator"] = create_eval_dataloader_iter(initial_app_state)

    with gr.Blocks(title="LM Component Activation Explorer") as app:
        # --- State Management ---
        # Store heavy objects and mutable state here
        app_state = gr.State(initial_app_state)
        # REMOVE: dataloader_iter = gr.State(initial_dataloader_iter) # CANNOT STORE ITERATOR IN STATE
        # State for current prompt data
        current_input_ids = gr.State(None)
        current_token_map = gr.State(None)  # Stores the TokenMap list
        current_masks = gr.State(None)
        # State for user selections
        selected_token_info = gr.State(None)  # Stores the selected TokenMapItem
        selected_layer_name = gr.State(
            initial_app_state["target_layer_names"][0]
            if initial_app_state["target_layer_names"]
            else None
        )

        # --- UI Layout ---
        gr.Markdown(f"# LM Component Activation Explorer\n**Model:** {args.model_path}")

        with gr.Row():
            # Use Textbox for raw display, HighlightedText for interaction
            # prompt_display_text = gr.Textbox(label="Prompt Text", lines=5, interactive=False)
            prompt_display_highlight = gr.HighlightedText(
                label="Prompt (Click Tokens)",
                interactive=True,
                combine_adjacent=False,  # Treat each token span separately
                show_legend=False,
            )
            next_button = gr.Button("Load Next Prompt")

        layer_dropdown = gr.Dropdown(
            label="Select Layer",
            choices=initial_app_state["target_layer_names"],
            value=initial_app_state["target_layer_names"][0]
            if initial_app_state["target_layer_names"]
            else None,
            interactive=True,
        )

        # Initially hidden area for token details
        with gr.Column(visible=False) as token_info_area:
            token_summary = gr.Markdown()
            active_components_summary = gr.Markdown()
            active_indices_table = gr.DataFrame(
                headers=["Component Index"],
                datatype=["number"],
                col_count=(1, "fixed"),
                # max_rows=10,  # Limit visible rows initially
                interactive=False,
                label="Active Component Indices",
            )
            future_plots_placeholder = gr.Markdown("*Future analyses will appear here.*")

        # --- Event Handlers ---

        def load_next_prompt_data(
            app_state_val: AppState,  # Removed current_iter_state input
        ) -> tuple[
            # REMOVED: Iterator,  # Updated iterator state
            Int[Tensor, "1 seq_len"],  # current_input_ids
            TokenMap,  # current_token_map
            dict[str, Float[Tensor, "1 seq_len m"]],  # current_masks
            list[tuple[str, str | None]],  # HighlightedText value
            # Reset selections
            None,  # selected_token_info
            str,  # token_summary update
            str,  # active_comp_summary update
            None,  # active_indices_table update
            gr.update,  # Return an update for the Column
        ]:
            """Gets next batch, calculates masks, prepares display data."""
            logger.info("Attempting to load next prompt...")
            # Access the iterator from the global store
            iterator = global_iterator_store["iterator"]
            if iterator is None:
                logger.error("Iterator not initialized.")
                raise gr.Error("Iterator not initialized. Please restart the app.")

            try:
                batch = next(iterator)
                input_ids: Int[Tensor, "1 seq_len"] = batch[
                    "input_ids"
                ]  # Should be shape (1, seq_len)
                logger.info(f"Loaded batch with shape: {input_ids.shape}")
            except StopIteration:
                logger.warning("Dataloader iterator exhausted. Resetting.")
                # Recreate iterator and update the global store
                iterator = create_eval_dataloader_iter(app_state_val)
                global_iterator_store["iterator"] = iterator
                try:
                    batch = next(iterator)
                    input_ids = batch["input_ids"]
                    logger.info(f"Loaded first batch after reset, shape: {input_ids.shape}")
                except StopIteration:
                    logger.error("Failed to get data even after resetting dataloader.")
                    # Handle error state appropriately, maybe raise gr.Error
                    raise gr.Error("Dataset seems empty or failed to load.")

            # Ensure input_ids are on CPU for mapping
            input_ids_cpu = input_ids.cpu()
            tokenizer = app_state_val["tokenizer"]

            # 1. Get Token Mapping
            full_text, token_map = get_token_mapping(tokenizer, input_ids_cpu)
            # Format for HighlightedText: list of (text_substring, label/tooltip)
            # We use the token index as the label for now.
            highlight_data = [(item["text"], f"Token {item['index']}") for item in token_map]

            # 2. Calculate Masks (can run on GPU if available)
            masks = calculate_masks_for_batch(
                app_state_val, input_ids
            )  # input_ids passed might be on CPU or GPU based on device

            # 3. Return all updates (excluding the iterator)
            return (
                # REMOVED: current_iter_state,
                input_ids_cpu,  # Store CPU version in state
                token_map,
                masks,
                highlight_data,
                None,  # Reset selected token
                "",  # Reset token_summary
                "",  # Reset active_components_summary
                None,  # Reset active_indices_table
                gr.update(visible=False),  # Return update object
            )

        next_button.click(
            fn=load_next_prompt_data,
            inputs=[app_state],  # Removed dataloader_iter
            outputs=[
                # REMOVED: dataloader_iter,
                current_input_ids,
                current_token_map,
                current_masks,
                prompt_display_highlight,
                selected_token_info,  # Reset
                token_summary,
                active_components_summary,
                active_indices_table,
                token_info_area,  # Reset
            ],
            queue=True,  # Use queue for potentially long-running model calls
        )

        def handle_token_select(
            evt: gr.SelectData,
            current_token_map_val: TokenMap | None,
            current_masks_val: dict[str, Float[Tensor, "1 seq_len m"]] | None,
            selected_layer_name_val: str | None,
        ) -> tuple[TokenMapItem | None, str, str, Any, gr.update]:
            """Handles token selection, finds corresponding token info, updates display."""
            logger.debug(f"Token selected event: Index={evt.index}, Value='{evt.value}'")
            selected_info = None
            if current_token_map_val:
                # Find the token corresponding to the selected character span
                char_index = evt.index[0]  # Start index of the selection
                for item in current_token_map_val:
                    start, end = item["span"]
                    # Check if the click falls within this token's span
                    if start <= char_index < end:
                        selected_info = item
                        logger.info(
                            f"Mapped selection at char {char_index} to token index {selected_info['index']}"
                        )
                        break
                if not selected_info:
                    logger.warning(f"Could not map character index {char_index} to any token span.")

            # Update the display based on the found token and current layer
            token_sum, active_sum, active_table, visible = update_token_display(
                selected_info, selected_layer_name_val, current_masks_val
            )
            return selected_info, token_sum, active_sum, active_table, gr.update(visible=visible)

        prompt_display_highlight.select(
            fn=handle_token_select,
            inputs=[current_token_map, current_masks, selected_layer_name],
            outputs=[
                selected_token_info,
                token_summary,
                active_components_summary,
                active_indices_table,
                token_info_area,
            ],
            queue=False,  # Selection should be fast
        )

        def handle_layer_change(
            layer_name: str,
            selected_token_info_val: TokenMapItem | None,
            current_masks_val: dict[str, Float[Tensor, "1 seq_len m"]] | None,
        ) -> tuple[str, str, Any, gr.update]:
            """Handles layer change, updates display if a token is selected."""
            logger.info(f"Layer changed to: {layer_name}")
            # Update display based on the new layer and existing token selection
            token_sum, active_sum, active_table, visible = update_token_display(
                selected_token_info_val, layer_name, current_masks_val
            )
            return token_sum, active_sum, active_table, gr.update(visible=visible)

        layer_dropdown.change(
            fn=handle_layer_change,
            inputs=[layer_dropdown, selected_token_info, current_masks],
            outputs=[
                token_summary,
                active_components_summary,
                active_indices_table,
                token_info_area,
            ],
            queue=False,  # Layer change update should be fast
        )

        # --- Initial Load ---
        # Use the same function, but ensure the global iterator is used
        app.load(
            fn=load_next_prompt_data,
            inputs=[app_state],
            outputs=[
                # REMOVED: dataloader_iter,
                current_input_ids,
                current_token_map,
                current_masks,
                prompt_display_highlight,
                selected_token_info,  # Reset
                token_summary,
                active_components_summary,
                active_indices_table,
                token_info_area,  # Reset - Keep this here
            ],
            queue=True,
        )

    return app


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradio app to explore LM component activations.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path or W&B reference to the trained SSModel. Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a publicly shareable link.",
    )
    args = parser.parse_args()

    gradio_app = build_gradio_app(args)
    # The iterator is initialized inside build_gradio_app now
    gradio_app.launch(share=args.share)
