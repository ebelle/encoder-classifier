import os
import time
import numpy as np
import torch
from utils import prep_batch, epoch_time


def train_step(
    model,
    source,
    src_len,
    targets,
    task,
    criterion,
    optimizer,
    clip,
    teacher_forcing=None,
):
    # source = [src_len, bsz]
    # targets = [trg_len, bsz]
    # src_len = [bsz]

    if task == "translation":
        output = model(source, src_len, targets, teacher_forcing)
    elif task == "tagging":
        output = model(source, src_len)
    # output = [src_len, bsz, output dim]

    output_dim = output.shape[-1]
    output = output[1:].view(-1, output_dim)
    # output = [(trg len - 1) * bsz, output dim]

    targets = targets[1:].view(-1)
    # targets = [(trg len - 1) * bsz]

    loss = criterion(output, targets)

    # delete variables to free up memory
    del output, targets

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()

    # get the value of the loss
    loss = loss.item()
    return loss


def train_model(
    model,
    iterator,
    task,
    optimizer,
    criterion,
    clip,
    device,
    epoch,
    start_time,
    save_path,
    dropout,
    pad_indices,
    num_batches,
    teacher_forcing=None,
    checkpoint=None,
    repr_layer=None,
):

    model.train()
    epoch_loss = 0
    batch_loss = []
    if task == "tagging":
        # save 10 times throughout training
        save_loss = np.linspace(0, num_batches, num=10, dtype=int)
    elif task == "translation":
        # save 100 times throughout training
        save_loss = np.linspace(0, num_batches, num=100, dtype=int)

    try:
        for i, batch in enumerate(iterator):
            source, targets, src_len = prep_batch(batch, device, pad_indices)
            optimizer.zero_grad()
            loss = train_step(
                model,
                source,
                src_len,
                targets,
                task,
                criterion,
                optimizer,
                clip,
                teacher_forcing,
            )

            epoch_loss += loss
            if i in save_loss:
                batch_loss.append(loss)
                end_time = time.time()

                batch_mins, batch_secs = epoch_time(start_time, end_time)

                print(
                    f"epoch {epoch} batch: {i} | Train loss: {loss:.3f} | Time: {batch_mins}m {batch_secs}s"
                )
                start_time = end_time

            # optionally checkpoint
            if checkpoint is not None:
                if i % checkpoint == 0:
                    adam, sparse_adam = optimizer.return_optimizers()
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "adam_state_dict": adam.state_dict(),
                            "sparse_adam_state_dict": sparse_adam.state_dict(),
                            "loss": loss,
                            "dropout": dropout,
                            "repr_layer": repr_layer,
                        },
                        os.path.join(save_path, f"checkpoint_{epoch}_{i}.pt"),
                    )
                    print(
                        f"Checkpoint saved at epoch {epoch} batch {i}. Train loss is {loss:.3f}"
                    )
    # skip batch in case of OOM
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"| WARNING: ran out of memory, skipping batch number {i:,}")

    return epoch_loss / num_batches, batch_loss
