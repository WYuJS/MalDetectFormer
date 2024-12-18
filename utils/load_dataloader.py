from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from sklearn.model_selection import train_test_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class TimeSeriesDataset(Dataset):
    def __init__(self, data, tmbed, labels):
        """
        初始化数据集。
        参数:
        data -- 数据的张量。
        tmbed -- 时间编码数据的张量。
        labels -- 标签的张量。
        """
        self.data = data
        self.tmbed = tmbed
        self.labels = labels

    def __len__(self):
        """
        返回数据集中的数据点数量。
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        返回索引为idx的数据点。
        """
        data_point = self.data[idx]
        tmbed_point = self.tmbed[idx]
        label = self.labels[idx]
        return data_point, tmbed_point, label

def load_dataloader(data, labels, tmbed, args):
    if tmbed == None:
        tmbed = torch.zeros(size=data.shape).to(device)
    X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=(args.test_set + args.val_set),
                                                        shuffle=False)
    t_train, t_temp = train_test_split(tmbed, test_size=(args.test_set + args.val_set),
                                                        shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp,
                                                    test_size=(args.test_set / (args.test_set + args.val_set)),
                                                    shuffle=False)
    t_val, t_test = train_test_split(t_temp, test_size=(args.test_set / (args.test_set + args.val_set)),
                                                    shuffle=False)
    train_dataset = TimeSeriesDataset(X_train, t_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, t_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, t_test, y_test)

    # test_dataset = TimeSeriesDataset(X_test, t_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_dataloader, val_dataloader, test_dataloader