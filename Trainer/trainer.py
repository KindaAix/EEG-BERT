# trainer.py
import os
import json
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from Logger.logger import Logger
from models.metrics import Metrics
from uilts.Visualizer import MetricsVisualizer
from models.loss import EEGLoss


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader,
                 criterion=None, optimizer=None, scheduler=None,
                 device="cuda", res="res", log_name="train", config=None):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.MLM_Loss, self.ESS_Loss = criterion
        adamw = torch.optim.AdamW(self.model.parameters())
        self.optimizer = optimizer if optimizer is not None else adamw
        self.scheduler = scheduler

        self.metrics = Metrics()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.exp_dir = os.path.join(res, f"exp-{timestamp}")  # exp path
        self.log_dir = os.path.join(self.exp_dir, "logs")  # log path
        self.ckpt_dir = os.path.join(self.exp_dir, "checkpoints")  # checkpoint path
        self.summary_dir = os.path.join(self.exp_dir, "summary")  # summary path

        for d in [self.log_dir, self.ckpt_dir, self.summary_dir]:
            os.makedirs(d, exist_ok=True)

        cfg_path = os.path.join(self.exp_dir, "config.json")  # save config
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4)
        self.logger = Logger(save_dir=self.log_dir, log_name=log_name)
        self.visualizer = MetricsVisualizer(save_dir=self.summary_dir)

        self.save_dir = self.ckpt_dir

        self.history = {
            "train": {"loss": [], "accuracy": [], "top_2_accuracy": []},
            "val": {"loss": [], "accuracy": [], "precision": [], "recall": [], "f1": [], "top_2_accuracy": []}
        }

    def _run_epoch(self, loader, train=True):
        """单个 epoch 的训练/验证逻辑"""
        epoch_loss, all_logits, all_labels = 0, [], []
        self.model.train() if train else self.model.eval()

        loop = tqdm(loader, desc="Train" if train else "Val", leave=False)
        for x, y in loop:
            x, y = x.to(self.device), y.to(self.device)
            if train:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                logits = self.model(x)
                loss = self.criterion(logits, y)
                if train:
                    loss.backward()
                    self.optimizer.step()

            epoch_loss += loss.item() * x.size(0)
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.detach().cpu())

        avg_loss = epoch_loss / len(loader.dataset)
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        if train:
            metrics = self.metrics.train_metrics(all_logits, all_labels)
        else:
            metrics = self.metrics.val_metrics(all_logits, all_labels)

        metrics["loss"] = avg_loss
        return metrics

    def train(self, num_epochs=50, early_stop=10):
        best_val_f1, patience = 0, 0

        for epoch in range(1, num_epochs + 1):
            self.logger.info(f"Epoch {epoch}/{num_epochs}")

            train_metrics = self._run_epoch(self.train_loader, train=True)
            val_metrics = self._run_epoch(self.val_loader, train=False)

            # 保存历史
            for k, v in train_metrics.items():
                if k in self.history["train"]:
                    self.history["train"][k].append(v)
            for k, v in val_metrics.items():
                if k in self.history["val"]:
                    self.history["val"][k].append(v)

            # Logger
            self.logger.info(f"Train: {train_metrics}")
            self.logger.info(f"Val:   {val_metrics}")
            for k, v in train_metrics.items():
                self.logger.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_metrics.items():
                self.logger.add_scalar(f"val/{k}", v, epoch)


            # Scheduler
            if self.scheduler:
                self.scheduler.step(val_metrics["loss"])


            # save *.pt of model
            if epoch // 20 == 0:
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, f"epoch_{epoch}.pt"))
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                patience = 0
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, "best_model.pt"))
                self.logger.info("Best model saved.")
            else:
                patience += 1
                if patience >= early_stop:
                    self.logger.info("Early stopping triggered.")
                    break

        # 保存指标历史为 CSV
        self._save_history()

        # 绘制曲线
        self.visualizer.plot_train_val(self.history["train"], self.history["val"])

    def test(self):
        self.model.load_state_dict(torch.load(os.path.join(self.ckpt_dir, "best_model.pt"), map_location=self.device))
        self.model.eval()

        all_logits, all_labels = [], []
        with torch.no_grad():
            for x, y in tqdm(self.test_loader, desc="Test"):
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                all_logits.append(logits.cpu())
                all_labels.append(y.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        test_metrics = self.metrics.test_metrics(all_logits, all_labels)
        self.logger.info(f"Test results: {test_metrics}")
        self.visualizer.plot_test(test_metrics)
        return test_metrics

    def _save_history(self):
        """保存训练过程指标为 CSV"""
        for phase in ["train", "val"]:
            df = pd.DataFrame(self.history[phase])
            df.to_csv(os.path.join(self.log_dir, f"{phase}_history.csv"), index=False)

