import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

dist.init_process_group("cpu:gloo,cuda:nccl")

default_group = dist.group.WORLD
default_backend = dist.get_backend(default_group)
default_ranks = list(range(dist.get_world_size()))

# create additional process group for asynchronous collectives
checkpoint_group = dist.new_group(default_ranks, backend=default_backend)

future_obj = None
for epoch in range(NUM_EPOCHS):
    torch.manual_seed(epoch)
    x, y = _input()
    loss = loss_calc(model(x), y)

    loss.backward()
    optim.step()
    optim.zero_grad()

    if epoch % SAVE_PERIOD == 0:
        if future_obj is not None:
            future_obj.result()
        future_obj = dcp.state_dict_save.async_save(
            state_dict,
            checkpoint_id = CHECKPOINT_DIR,
            process_group = checkpoint_group,
        )
