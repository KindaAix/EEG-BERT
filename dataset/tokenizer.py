import os
import numpy as np
import scipy.io as sio

# ----------------每个数据对应4秒----------------

# -------------------- 配置 --------------------
data_dir = 'dataset/EEG-FEATURE'
save_dir = 'dataset/EEG-Tokens'
score_file_dir = 'dataset/continuous_labels'  # 连续标签分数文件夹
window_size = 10
step_size = 1
threshold = 15

# 假设每个 mat 文件有 80 个 trial，标签文件用列表存储，和 mat 文件同名
labels_lst = [
    "Happy", "Neutral", "Disgust", "Sad", "Anger", "Anger", "Anger", "Sad", "Disgust", "Neutral", "Happy", "Happy", "Neutral", "Disgust", "Sad", "Anger", "Anger", "Sad", "Disgust", "Neutral", "Happy",
    "Anger", "Sad", "Fear", "Neutral", "Surprise", "Surprise", "Neutral", "Fear", "Sad", "Anger", "Anger", "Sad", "Fear", "Neutral", "Surprise", "Surprise", "Neutral", "Fear", "Sad", "Anger",
    "Happy", "Surprise", "Disgust", "Fear", "Anger", "Anger", "Anger", "Fear", "Disgust", "Surprise", "Happy", "Happy", "Surprise", "Disgust", "Fear", "Anger", "Anger", "Fear", "Disgust", "Surprise", "Happy",
    "Disgust", "Sad", "Fear", "Surprise", "Happy", "Happy", "Surprise", "Fear", "Sad", "Disgust", "Sad", "Fear", "Surprise", "Happy", "Happy", "Surprise", "Fear", "Sad", "Disgust"
]
labels_dict = {
    "Disgust": 0,
    "Fear": 1,
    "Sad": 2,
    "Neutral": 3,
    "Happy": 4,
    "Anger": 5,
    "Surprise": 6
}
# -------------------- Token 切分函数 --------------------
def trial_to_tokens(trial, cont_scores, window_size=10, step_size=1, threshold=15):
    """
    trial: numpy array, shape (L, 62, 5)
    cont_scores: numpy array, shape (L,)  # 该 trail 对应的连续标签分数
    window_size: 每个 token 的帧数
    step_size: 滑动窗口步长
    threshold: 平均分数阈值 (>=15 才保留)

    return: list of token, 每个 token shape = (window_size, 310)
    """
    tokens = []
    L = trial.shape[0]
    
    for start in range(0, L - window_size + 1, step_size):
        end = start + window_size

        # 计算该窗口的连续分数均值
        score_mean = np.mean(cont_scores[start:end])
        
        if score_mean >= threshold:
            token = trial[start:end]
            # 拉平成 (window_size, 310)，方便后续处理
            tokens.append(token.reshape(token.shape[0], -1))
    
    print(f"总共提取 **{len(tokens)} **个 token")
    
    return tokens


# -------------------- 处理 mat 文件 --------------------
mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]

for mat_file in mat_files:
    mat_num = os.path.splitext(mat_file)[0]  # "1.mat" -> "1"
    mat_path = os.path.join(data_dir, mat_file)
    score_path = os.path.join(score_file_dir, mat_file)
    mat_data = sio.loadmat(mat_path)

    # 提取所有 de_LDS_* 键
    de_lds_keys = [k for k in mat_data.keys() if k.startswith('de_LDS_')]
    de_lds_keys.sort(key=lambda x: int(x.split('_')[-1]))  # 按 trail 顺序

    for i, key in enumerate(de_lds_keys):
        trial = mat_data[key]
        trial = trial.transpose(0, 2, 1)  # shape: (L, 62, 5)
        cont_scores = np.squeeze(sio.loadmat(score_path)[str(i+1)])
        tokens = trial_to_tokens(trial, cont_scores, window_size, step_size, threshold)

        # 该 trial 的标签
        label = labels_dict[labels_lst[i]]
        for id, token in enumerate(tokens):
            # 保存成 npz 文件
            save_path = os.path.join(save_dir, f"trail_{mat_num}_{i+1}_{id+1}.npz")
            np.savez(save_path, data=token, label=label)
            print(f"保存 {save_path}, tokens shape={token.shape}, label={label}")

    