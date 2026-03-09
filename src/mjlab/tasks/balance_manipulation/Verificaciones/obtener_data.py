import wandb

api = wandb.Api()
run = api.run("/abadjosue25-abba/mjlab/runs/nbhhj68v")


print(run.history())
