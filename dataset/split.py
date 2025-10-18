import os
import shutil
import random
from tqdm import tqdm

def stratified_split_npz(npz_files, save_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    按比例划分 npz 文件到 train/val/test 文件夹，并计算类别权重
    Args:
        npz_files: 所有 npz 文件路径列表
        save_dir: 保存的目录
    """
    os.makedirs(save_dir, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(save_dir, split), exist_ok=True)

    # 标签字典
    labels_dict = {
        "Disgust": 0,
        "Fear": 1,
        "Sad": 2,
        "Neutral": 3,
        "Happy": 4,
        "Anger": 5,
        "Surprise": 6
    }

    labels_lst = [
        "Happy", "Neutral", "Disgust", "Sad", "Anger", "Anger", "Anger", "Sad", "Disgust", "Neutral", "Happy",
        "Anger", "Sad", "Fear", "Neutral", "Surprise", "Surprise", "Neutral", "Fear", "Sad", "Anger", "Anger", 
        "Sad", "Fear", "Neutral", "Surprise", "Surprise", "Neutral", "Fear", "Sad", "Anger", "Anger", "Sad", 
        "Fear", "Neutral", "Surprise", "Surprise", "Neutral", "Fear", "Sad", "Anger", "Happy", "Surprise", 
        "Disgust", "Fear", "Anger", "Anger", "Anger", "Fear", "Disgust", "Surprise", "Happy", "Happy", 
        "Surprise", "Disgust", "Fear", "Anger", "Anger", "Fear", "Disgust", "Surprise", "Happy", "Disgust", 
        "Sad", "Fear", "Surprise", "Happy", "Happy", "Surprise", "Fear", "Sad", "Disgust", "Sad", "Fear", 
        "Surprise", "Happy", "Happy", "Surprise", "Fear", "Sad", "Disgust"
    ]

    # 把样本分到各个类别
    labels = {key: [] for key in range(7)}
    for npz in tqdm(npz_files, desc='标签搜索...'):
        # 根据文件名中倒数第二个数字定位标签
        idx = int(npz.split('_')[-2])
        cls = labels_dict[labels_lst[idx]]
        labels[cls].append(npz)

    # 按比例划分并保存
    split_files = {"train": [], "val": [], "test": []}
    for cls, files in tqdm(labels.items(), desc='划分中...'):
        random.shuffle(files)
        n_total = len(files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        train_files = files[:n_train]
        val_files = files[n_train:n_train+n_val]
        test_files = files[n_train+n_val:]

        for split, fs in zip(["train", "val", "test"], [train_files, val_files, test_files]):
            target_dir = os.path.join(save_dir, split, str(cls))
            os.makedirs(target_dir, exist_ok=True)
            for f in fs:
                shutil.copy(f, os.path.join(target_dir, os.path.basename(f)))
                split_files[split].append(f)

    # 计算类别权重
    total = sum(len(v) for v in labels.values())
    num_classes = len(labels)
    class_weights = {cls: total/(num_classes*len(v)) for cls, v in labels.items()}

    print("类别数量:", {cls: len(v) for cls, v in labels.items()})
    print("类别权重:", class_weights)

    return split_files, class_weights


stratified_split_npz(
                    [os.path.join('dataset/EEG-Tokens', i) for i in os.listdir('dataset/EEG-Tokens')], 
                    'dataset/EEG-Tokens', 
                    train_ratio=0.7, 
                    val_ratio=0.15, 
                    test_ratio=0.15
                    )