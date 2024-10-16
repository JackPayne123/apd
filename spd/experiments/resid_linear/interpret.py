# %%

import torch
import torch.nn.functional as F

from spd.experiments.resid_linear.models import ResidualLinearModel, ResidualLinearSPDFullRankModel
from spd.experiments.resid_linear.resid_linear_dataset import ResidualLinearDataset
from spd.run_spd import ResidualLinearConfig, calc_recon_mse
from spd.utils import run_spd_forward_pass, set_seed

# %%

if __name__ == "__main__":
    # Set up device and seed
    device = "cpu"
    print(f"Using device: {device}")
    set_seed(0)  # You can change this seed if needed

    # wandb_path = "spd-resid-linear/runs/v7ajkx3j"
    # wandb_path = "spd-resid-linear/runs/lhqhwkzy" # Run with topk_recon=10000 with bad attrs
    # wandb_path = "spd-resid-linear/runs/xpd6towq"
    # wandb_path = "spd-resid-linear/runs/ppahfs1r"
    # wandb_path = "spd-resid-linear/runs/7ivgbm9q"
    # wandb_path = "spd-resid-linear/runs/1wrmt7dr"  # With identity W_E
    # wandb_path = "spd-resid-linear/runs/85w6rt69"  # With identity W_E and 5 features

    # wandb_path = "spd-resid-linear/runs/61ovotx2"  # With identity W_E and 2 features
    wandb_path = "spd-resid-linear/runs/7ijm2ece"  # With identity W_E and 2 features
    # local_path = "spd/experiments/resid_linear/out/fr_seed0_topk1.10e+00_topkrecon1.00e+00_topkl2_1.00e-02_lr1.00e-02_bs1024_ft5_lay1_resid5_mlp5/model_10000.pth"

    # Load the pretrained SPD model
    model, config, label_coeffs = ResidualLinearSPDFullRankModel.from_wandb(wandb_path)
    # model, config, label_coeffs = ResidualLinearSPDFullRankModel.from_local_path(local_path)

    assert isinstance(config.task_config, ResidualLinearConfig)
    # Path must be local
    target_model, target_config_dict, target_label_coeffs = ResidualLinearModel.from_pretrained(
        config.task_config.pretrained_model_path
    )
    assert target_label_coeffs == label_coeffs

    dataset = ResidualLinearDataset(
        embed_matrix=model.W_E,
        n_features=model.n_features,
        feature_probability=config.task_config.feature_probability,
        device=device,
        label_coeffs=label_coeffs,
        one_feature_active=config.task_config.one_feature_active,
    )
    batch, labels = dataset.generate_batch(config.batch_size)
    # Print some basic information about the model
    # print(f"Model structure:\n{model}")
    print(f"Number of features: {model.n_features}")
    print(f"Embedding dimension: {model.d_embed}")
    print(f"MLP dimension: {model.d_mlp}")
    print(f"Number of layers: {model.n_layers}")
    print(f"Number of subnetworks (k): {model.k}")

    assert config.topk is not None
    spd_outputs = run_spd_forward_pass(
        spd_model=model,
        target_model=target_model,
        input_array=batch,
        full_rank=config.full_rank,
        ablation_attributions=config.ablation_attributions,
        batch_topk=config.batch_topk,
        topk=config.topk,
        distil_from_target=config.distil_from_target,
    )
    # Topk recon (Note that we're using true labels not the target model output)
    topk_recon_loss = calc_recon_mse(spd_outputs.spd_topk_model_output, labels)
    print(f"Topk recon loss: {topk_recon_loss}")
    print(f"batch:\n{batch[:10]}")
    print(f"labels:\n{labels[:10]}")
    print(f"spd_outputs.spd_topk_model_output:\n{spd_outputs.spd_topk_model_output[:10]}")
    # print(f"spd_model_output:\n{spd_outputs.spd_model_output[:10]}")

    in_matrix = model.W_E @ target_model.layers[0].input_layer.weight.T
    print(f"target in_matrix:\n{in_matrix}")

    in_matrix_subnet0 = model.W_E @ model.layers[0].linear1.subnetwork_params[0]
    print(f"in_matrix_subnet0:\n{in_matrix_subnet0}")
    in_matrix_subnet1 = model.W_E @ model.layers[0].linear1.subnetwork_params[1]
    print(f"in_matrix_subnet1:\n{in_matrix_subnet1}")

    out_matrix = target_model.layers[0].output_layer.weight.T
    print(f"target out_matrix:\n{out_matrix}")
    out_matrix_subnet0 = model.layers[0].linear2.subnetwork_params[0]
    print(f"out_matrix_subnet0:\n{out_matrix_subnet0}")
    out_matrix_subnet1 = model.layers[0].linear2.subnetwork_params[1]
    print(f"out_matrix_subnet1:\n{out_matrix_subnet1}")
# %%
if __name__ == "__main__":
    # Load the target model
    # target_path = "spd/experiments/resid_linear/out/resid_linear_n-features5_d-resid5_d-mlp5_n-layers1_seed0/model.pth"
    target_path = "spd/experiments/resid_linear/out/resid_linear_n-features2_d-resid2_d-mlp2_n-layers1_seed0/target_model.pth"
    model, config, label_coeffs = ResidualLinearModel.from_pretrained(target_path)

    # Make a simple batch which is just a value of 0.5
    dataset = ResidualLinearDataset(
        embed_matrix=model.W_E,
        n_features=2,
        feature_probability=0.5,
        device="cpu",
        label_coeffs=label_coeffs,
    )

    print(f"Label coeffs: {label_coeffs}")
    batch, label = dataset.generate_batch(1)
    print(f"Batch: {batch}")
    # Apply Gelu(ax) + x to batch
    raw_labels = F.gelu(batch * torch.tensor(label_coeffs)) + batch
    print(f"Raw labels: {raw_labels}")
    embedded_labels = raw_labels @ model.W_E
    print(f"Embedded labels: {embedded_labels}")

    # Forward pass
    outputs = model(batch)
    print(f"Outputs: {outputs}")
    print(f"Label: {label}")

# %%
