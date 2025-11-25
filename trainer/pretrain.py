import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from models.pretrain_model import BERTLM
from models.bert import BERT
from optim_schedule import ScheduledOptim

import tqdm

"""
mask_function: random mask input tensor
kmeans: kmeans model for clustering the input features
random single for continuous masking
loss function: cross entropy loss for classification tasks
"""

class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.

        1. Channel Masked Language Model : 3.3.1 Task #1: Masked LM
        2. Emotional Sentence Segmentation : 3.3.2 Task #2: Emotional Sentence Segmentation

    please check the details on README.md with simple example.

    """

    def __init__(self, bert: BERT,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert
        # Initialize the BERT Language Model, with BERT model
        self.model = BERTLM(bert).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.bert.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        """"""
        self.criterion = nn.NLLLoss(ignore_index=0)  # 损失函数需要换成交叉熵损失

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu), bert_input must be masked before sent to the model
            data["bert_input"] = self.random_mask(data["bert_input"]).to(self.device)  # masked input [B, seq_len, n_features]
            data["is_change"] = data["is_change"].to(self.device)
            cluster_label_1st = self.kmeans().predict(data["bert_input"].reshape(-1, data["bert_input"].shape[-1])).to(self.device)  # [B*seq_len, n_features]
            # 1. forward the ess and cmlm model
            ess_output, cmlm_output, aux_loss, _ = self.model.forward(data["bert_input"], aux_loss=0.0)

            # 2-1. CE loss of is_change classification result
            ess_loss = self.criterion(ess_output, data["is_change"])

            # 2-2. CE loss of predicting masked token cluster id
            cmlm_loss = self.criterion(cmlm_output, cluster_label_1st)

            # 2-3. Adding ess_loss, mask_loss and aux_loss : 3.4 Pre-training Procedure
            loss = ess_loss + cmlm_loss + aux_loss

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                # re-update the parameters of the model           
                ess_output, cmlm_output, aux_loss, feature = self.model.forward(data["bert_input"], aux_loss=0.0)
                cluster_label_2nd = self.kmeans().predict(feature.reshape(-1, data["bert_input"].shape[-1])).to(self.device)  # [B*seq_len, embedding_size]
                ess_loss = self.criterion(ess_output, data["is_change"])
                cmlm_loss = self.criterion(cmlm_output, cluster_label_2nd)
                loss = ess_loss + cmlm_loss + aux_loss

                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()
            # next sentence prediction accuracy
            correct = ess_output.argmax(dim=-1).eq(data["is_change"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["is_change"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=",
              total_correct * 100.0 / total_element)

    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
