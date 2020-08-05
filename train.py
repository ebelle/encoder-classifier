import os
import time
import torch
from utils import prep_batch, epoch_time


def train_nmt_step(
    model, source, src_len, targets, criterion, optimizer, clip, teacher_forcing,
):
    # source = [src_len, bsz]
    # targets = [trg_len, bsz]
    # src_len = [bsz]

    output = model(source, src_len, targets, teacher_forcing)
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

    # extract the loss value but don't hold the tensors in memory
    loss = loss.item()
    return loss


def train_nmt_model(
    model,
    iterator,
    optimizer,
    criterion,
    clip,
    teacher_forcing,
    device,
    epoch,
    start_time,
    save_path,
    checkpoint=None,
):

    model.train()
    epoch_loss = 0
    print(f"len iterator: {len(iterator)}")
    for i, batch in enumerate(iterator):
        source, targets, src_len = prep_batch(batch, device)
        optimizer.zero_grad()
        try:
            loss = train_nmt_step(
                model,
                source,
                src_len,
                targets,
                criterion,
                optimizer,
                clip,
                teacher_forcing,
            )
            del source, targets, src_len
            epoch_loss += loss
            # shitty progress bar of sorts
            if i != 0 and i % 200 == 0:
                end_time = time.time()

                batch_mins, batch_secs = epoch_time(start_time, end_time)
                print(
                    f"batch: {i} | Train loss: {loss:.3f} | Time: {batch_mins}m {batch_secs}s"
                )
                start_time = end_time
            # optionally checkpoint
            if i != 0 and checkpoint is not None:
                if i % checkpoint == 0:
                    adam, sparse_adam = optimizer.return_optimizers()
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "adam_state_dict": adam.state_dict(),
                            "sparse_adam_state_dict": sparse_adam.state_dict(),
                            "loss": loss,
                        },
                        os.path.join(save_path, f"checkpoint_{epoch}_{batch}.pt"),
                    )
                    print(
                        f"Checkpoint saved at epoch {epoch} batch {i}. Train loss is {loss:.3f}"
                    )
        # skip batch in case of OOM
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"| WARNING: ran out of memory, skipping batch number {i:,}")

    return epoch_loss / len(iterator)