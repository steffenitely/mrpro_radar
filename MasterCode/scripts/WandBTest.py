import wandb
import time
import random

# Initialize a run
wandb.init(
    project="wandb-test-project",
    name="simple-test-run"
)

# Log some dummy data
for step in range(10):
    loss = random.random()
    accuracy = random.random()

    wandb.log({
        "step": step,
        "loss": loss,
        "accuracy": accuracy
    })

    print(f"Step {step}: loss={loss:.3f}, acc={accuracy:.3f}")
    time.sleep(0.5)

# Finish run
wandb.finish()