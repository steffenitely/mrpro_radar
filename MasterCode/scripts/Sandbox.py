import wandb
run = wandb.init(
    entity="mara-guastini-",
    project="estimate-lambdas",
)
run.config["params_motion_net"] = params_motion_net
run.config["params_reconstruction"] = params_reconstruction