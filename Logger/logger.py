# utils/logger.py
import logging
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, save_dir="logs", log_name="train"):
        """
        日志器
        :param save_dir: 日志保存目录
        :param log_name: 日志文件名前缀
        """
        os.makedirs(save_dir, exist_ok=True)
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")

        # 日志文件路径
        log_file = os.path.join(save_dir, f"{log_name}_{time_str}.log")

        # ---------- logging 配置 ----------
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.DEBUG)

        # 文件日志
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)

        # 控制台日志
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 日志格式
        formatter = logging.Formatter(
            "[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 避免重复添加 handler
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

        # ---------- TensorBoard ----------
        tb_dir = os.path.join(save_dir, "tensorboard", time_str)
        self.writer = SummaryWriter(log_dir=tb_dir)

        self.logger.info(f"Logger initialized. Log file: {log_file}")

    def info(self, msg: str):
        """记录普通信息"""
        self.logger.info(msg)

    def warning(self, msg: str):
        """记录警告信息"""
        self.logger.warning(msg)

    def error(self, msg: str):
        """记录错误信息"""
        self.logger.error(msg)

    def add_scalar(self, tag: str, value, step: int):
        """记录标量 (loss, acc 等)"""
        self.writer.add_scalar(tag, value, step)

    def add_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """同时记录多个标量"""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def close(self):
        self.writer.close()
