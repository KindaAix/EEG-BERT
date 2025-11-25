from transformers import Trainer
from tqdm import tqdm
import torch
from models.kmeans import torch_kmeans
from models.loss import EEGLoss


class EEGTrainer(Trainer):
    def __init__(self, *args, compute_loss_func=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute_loss_func = compute_loss_func

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        if self.compute_loss_func is not None:
            loss = self.compute_loss_func(outputs, inputs)
        else:
            loss = outputs.loss
        return (loss, outputs) if return_outputs else loss