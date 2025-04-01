from spd.experiments.tms.models import TMSSPDModel

if __name__ == "__main__":
    # run_id = "wandb:spd-tms/runs/u359w3kq"
    # run_id = "wandb:spd-tms/runs/hrwrgei2"
    run_id = "wandb:spd-tms/runs/3p8qgg6b"
    # pretrained_model_path = "wandb:spd-train-tms/runs/tmzweoqk"
    # run_id = "wandb:spd-tms/runs/fj68gebo"
    # target_model, target_model_train_config_dict = TMSModel.from_pretrained(pretrained_model_path)
    spd_model, spd_model_train_config_dict = TMSSPDModel.from_pretrained(run_id)

    pass
    # # We used "-" instead of "." as module names can't have "." in them
    # gates = {k.removeprefix("gates.").replace("-", "."): v for k, v in spd_model.gates.items()}

    # input_magnitude = 0.75
    # fig = plot_mask_vals(
    #     spd_model,
    #     target_model,
    #     gates,  # type: ignore
    #     device="cpu",
    #     input_magnitude=input_magnitude,
    # )
    # fig.savefig(f"tms_mask_vals_{input_magnitude}.png")
    # print(f"Saved figure to tms_mask_vals_{input_magnitude}.png")
