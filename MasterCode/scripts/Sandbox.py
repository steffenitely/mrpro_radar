import wandb

print("1 - importing wandb")

run = wandb.init(
    project="wandb-test",
    settings=wandb.Settings(
        init_timeout=20
    )
)

print("2 - wandb.init finished")

wandb.log({"test": 1})

print("3 - log finished")

run.finish()

print("4 - finished")