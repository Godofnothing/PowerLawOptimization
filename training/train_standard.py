import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Iterable, Callable, Optional

from wandb import wandb


def train_epoch(
    model : nn.Module, 
    dataloader : Iterable,
    optimizer : torch.optim.Optimizer,
    device: str = 'cpu'
):
    model.train()
    
    running_loss, running_correct = 0.0, 0.0
    
    for samples, labels in dataloader:
        samples, labels = samples.to(device), labels.to(device)
        # convert labels to one_hot
        one_hot = F.one_hot(labels, num_classes=10).to(samples.dtype)
        # get model output
        logits = model(samples)
        # compute loss
        loss = F.mse_loss(logits, one_hot)
        # make gradient step  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # get predictions from logits
        preds = torch.argmax(logits, dim=1)  
        # compute num of correct
        correct = torch.sum(preds == labels)
        # update running loss and accuracy
        running_loss += len(samples) * loss.item()
        running_correct += correct.item()
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_correct / len(dataloader.dataset)
    
    return {"loss" : epoch_loss, "acc" : epoch_acc}


@torch.no_grad()
def val_epoch(
    model : nn.Module, 
    dataloader : Iterable,
    device: str = 'cpu'
):
    model.eval()
    
    running_loss, running_correct = 0.0, 0.0
    
    for samples, labels in dataloader:
        samples, labels = samples.to(device), labels.to(device)
        # convert labels to one_hot
        one_hot = F.one_hot(labels, num_classes=10).to(samples.dtype)
        # get model output
        logits = model(samples)
        # compute loss
        loss = F.mse_loss(logits, one_hot)
        # get predictions from logits
        preds = torch.argmax(logits, dim=1)  
        # compute num of correct
        correct = torch.sum(preds == labels)
        # update running loss and accuracy
        running_loss += len(samples) * loss.item()
        running_correct += correct.item()
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_correct / len(dataloader.dataset)
    
    return {"loss" : epoch_loss, "acc" : epoch_acc}


def train(
    model : nn.Module, 
    dataloaders : Dict[str, Iterable], 
    optimizer : torch.optim.Optimizer,
    num_epochs: int, 
    scheduler : Optional[Callable] = None,
    device='cpu',
    log_frequency: int = 1,
    log_wandb: bool = False,
    save_dir: str = '',
):
    history = {
        "train" : {"loss" : [], "acc" : []},
        "val"   : {"loss" : [], "acc" : []}
    }

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # run train epoch
        train_stats = train_epoch(model, dataloaders["train"], optimizer, device=device)
        history["train"]["loss"].append(train_stats['loss'])
        history["train"]["acc"].append(train_stats['acc'])
        
        if epoch % log_frequency == 0:
            print(f"Epoch {epoch}/{num_epochs - 1}")
            print("-" * 10)
            print(f"{'Train':>5} Loss: {train_stats['loss']:.4f} Acc: {train_stats['acc']:.4f}")
            if log_wandb:
                wandb.log({"train/loss" : train_stats['loss'], "train/acc" : train_stats['acc']}, step=epoch)
        
        # run validation epoch
        if dataloaders.get("val"):
            val_stats = val_epoch(model, dataloaders["val"], device=device)
            history["val"]["loss"].append(val_stats['loss'])
            history["val"]["acc"].append(val_stats['acc'])
            if epoch % log_frequency == 0:
                print(f"{'Val':>5} Loss: {val_stats['loss']:.4f} Acc: {val_stats['acc']:.4f}")
                if log_wandb:
                    wandb.log({"val/loss" : val_stats['loss'], "val/acc" : val_stats['acc']}, step=epoch)

            if val_stats["acc"] > best_val_acc:
                best_val_acc = val_stats["acc"]
                # save best model
                if save_dir:
                    torch.save({
                        "epoch" : epoch,
                        "model_state_dict" : model.state_dict(),
                        "optimizer_state_dict" : optimizer.state_dict()
                    }, f"{save_dir}/best.pt")

        if scheduler:
            scheduler.step()

    # save last model
    if save_dir:
        torch.save({
            "epoch" : epoch,
            "model_state_dict" : model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict()
        }, f"{save_dir}/last.pt")
                
    return history
