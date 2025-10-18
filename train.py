# train.py
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 屏蔽 TF 的所有日志
import json
import torch
import argparse
from models.crnn import EEG_RCNN
from models.loss import EEGLoss
from Trainer.load_data import get_train_loader, get_valid_loader, get_test_loader
from Trainer.trainer import Trainer




def get_config():
    parser = argparse.ArgumentParser(description="Train EEG RCNN")

    # ========= 基础参数 =========
    parser.add_argument('--config', type=str, default='config.json', help='配置文件路径')
    parser.add_argument('--data_root', type=str, default=None, help='数据目录')
    parser.add_argument('--save_dir', type=str, default=None, help='模型保存目录')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None, help='批大小')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument("--kmeans_path", type=str, default=None, help='Torch-KMeans权重文件')

    # ========= 解析命令行参数 =========
    args = parser.parse_args()

    # ========= 读取配置文件 =========
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            cfg = json.load(f)    
    else:
        cfg = {}

    # ========= 合并参数（命令行优先）=========
    config = {}
    config['data_root']    = args.data_root   or cfg.get('data_root', 'dataset/')
    config['save_dir']    = args.save_dir   or cfg.get('save_dir', 'checkpoints/')
    config['epochs']      = args.epochs     or cfg.get('epochs', 50)
    config['batch_size']  = args.batch_size or cfg.get('batch_size', 32)
    config['lr']          = args.lr         or cfg.get('lr', 1e-3)
    # config['kmeans_path'] = args.kmeans     or cfg.get('kmeans_path', 'KMeans.pt')
    cfg.update(config)

    return cfg

def main():
    cfg = get_config()

    # dataloader
    train_loader = get_train_loader(data_root=cfg["data_root"], kmeans_path=cfg["kmeans_path"],
                                    batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])
    val_loader = get_valid_loader(data_root=cfg["data_root"], kmeans_path=cfg["kmeans_path"],
                                  batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])
    test_loader = get_test_loader(data_root=cfg["data_root"], kmeans_path=cfg["kmeans_path"],
                                  batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])

    # model
    model = EEG_RCNN(
        n_clusters=cfg["n_clusters"],
        embed_dim=cfg["embed_dim"],
        n_classes=cfg["n_classes"],
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )

    # loss
    criterion = EEGLoss(
        n_classes=cfg["n_classes"],
        class_weights=cfg.get("class_weights"),
        label_smoothing=cfg.get("label_smoothing", 0.0),
    )

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)

    # trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=cfg["device"],
        res=cfg["save_dir"],
        log_name=cfg["log_name"]
    )

    # train & test
    trainer.train(num_epochs=cfg["epochs"], early_stop=cfg["early_stop"])
    test_metrics = trainer.test()


if __name__ == "__main__":
    main()
