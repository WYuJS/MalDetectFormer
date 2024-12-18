import numpy as np
import pandas as pd
import glob
import os
import torch
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from dateutil import parser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 检查一个字符串是否是IP地址的函数
def is_ip(s):
    if isinstance(s, str):
        parts = s.split('.')
        return len(parts) == 4 and all(part.isdigit() for part in parts)
    return False


# 分割IP地址的函数
def split_ip(ip_str):
    return list(map(int, ip_str.split('.')))


# 处理IP地址并重新组合数据
def iptolist(data):
    processed_data = []
    for row in data:
        new_row = []
        for item in row:
            if is_ip(item):
                # 分割IP地址并扩展到new_row中
                new_row.extend(split_ip(item))
            else:
                # 非IP地址，直接添加
                new_row.append(item)
        processed_data.append(new_row)
    return processed_data


# 转换数据
def convert_to_int(x):
    try:
        # 尝试直接转换为整数
        return int(x)
    except ValueError:
        # 如果遇到ValueError，可能是十六进制，尝试按十六进制转换
        return int(x, 16)


def ip_to_network_adjust_columns(data):
    """
    将数据中的IP地址转换为网段，并将这些网段插入到第三列和第四列。

    参数:
    - data: 一个包含IP地址的二维列表。

    返回:
    - 更新后的列表，其中新的两列已被加到第三列和第四列。
    """
    new_data = []
    for idx, row in enumerate(data):
        # 去除没有源ip和，目的ip的列
        try:
            network_1 = (row[0].rsplit('.', 1)[0] + ".0/24")
            network_2 = (row[1].rsplit('.', 1)[0] + ".0/24")


        except AttributeError:
            # pass
            network_1 = "0.0.0.0/24"
            network_2 = "0.0.0.0/24"
        new_row = row[:2].tolist() + [str(network_1), str(network_2)] + row[2:].tolist()
        new_data.append(new_row)
    return new_data


def try_convert(value):
    """尝试将值转换为数值类型，如果不成功则编码。"""
    encoder = LabelEncoder()
    try:
        # 尝试转换为浮点数
        float_val = float(value)
        # 如果是整数，进一步转换为整型
        if float_val.is_integer():
            return int(float_val)
        return float_val
    except ValueError:
        return encoder.fit_transform([value])[0]


def convert_time_to_tensor(time_series):
    time_data = pd.Series(pd.to_datetime(time_series, format='%m/%d/%Y %H:%M', errors='coerce'))

    # 使用前一个有效值填充NaT值
    time_data = time_data.ffill()

    # 确保在开始的位置没有NaT值，如果有，则使用后向填充（bfill）
    # 这是为了处理数据开头就是NaT的情况
    time_data = time_data.bfill()

    # 将时间特征组装成一个DataFrame
    df = pd.DataFrame({
        'month': time_data.dt.month,
        'day': time_data.dt.day,
        'weekday': time_data.dt.weekday,
        'hour': time_data.dt.hour,
        'minute': time_data.dt.minute
    })

    # 将DataFrame转换为Tensor
    time_tensor = torch.tensor(df.values).to(device)
    del time_data
    del df
    gc.collect()
    return time_tensor


# 获取UNSW_NB15数据
def getUNSWData(args, is_multi=False):
    # unsw文件是数值型的编号
    unsw_numeric_list = np.array(
        [7, 8, 9, 15, 16, 17, 18, 23, 24, 27, 28, 31, 32, 33, 34, 35, 38, 40, 41, 42, 43, 44, 45, 46,
         47]) - 1 + 2
    # 设定文件夹路径，例如 "./data/"
    folder_path = 'data/UNSW_NB15'

    # 使用glob.glob()匹配文件夹下的所有CSV文件
    # 这里假设CSV文件的扩展名是小写的".csv"，如果有大写的".CSV"，需要另外匹配或统一文件扩展名
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    # 按文件名排序，确保顺序读取
    csv_files.sort()

    print(csv_files)
    # 初始化一个空的DataFrame用于存储整合后的数据
    combined_data = []

    # 遍历文件列表，读取每个CSV文件，并将其追加到combined_df中
    train_data = []
    val_data = []
    test_data = []
    for idx, file in enumerate(csv_files):
        # 读取CSV文件
        df = pd.read_csv(file, encoding='ISO-8859-1')
        df.fillna(0, inplace=True)
        df.replace('-', -1, inplace=True)
        df.replace('', -2, inplace=True)
        df.replace(' ', -2, inplace=True)

        data_list = df.to_numpy()
        train, temp = train_test_split(data_list, test_size=(args.test_set + args.val_set),
                                       shuffle=False)
        val, test = train_test_split(data_list, test_size=(args.test_set / (args.test_set + args.val_set)),
                                     shuffle=False)
        if idx == 0:
            train_data = train
            val_data = val
            test_data = test
        else:
            train_data = np.vstack([train_data, train])
            val_data = np.vstack([val_data, val])
            test_data = np.vstack([test_data, test])

    combined_data = np.vstack([train_data, val_data, test_data])
    # 创建LabelEncoder
    encoder = LabelEncoder()

    # 获取异常值
    combined_data = np.array((combined_data))
    combined_data[:, [1, 2]] = combined_data[:, [2, 1]]  # 把源ip和目的ip放到前两列
    anomaly_labels = torch.tensor((combined_data[:, -1].astype(np.float32))).long().to(device)  # 是否是异常(0, 1)
    type_labels = torch.tensor(encoder.fit_transform(combined_data[:, -2].astype(str))).long().to(device)  # 是哪种异常
    combined_data = combined_data[:, :-2]  # 去除最后两列是最后的data

    # anomaly_labels.requires_grad = True
    # type_labels.requires_grad = True

    # 将ip转换成网段，添加到数据中，此时数据：【源IP地址，目的IP地址，源网段，目的网段，...】
    combined_data = ip_to_network_adjust_columns(combined_data)
    combined_data = np.array(combined_data)
    # 分别对每列进行处理
    encoded_data = []
    numeric_data = []
    for i in range(len(combined_data[0])):
        column = combined_data[:, i]
        if not (i in unsw_numeric_list):  # 标签编码非数值列
            encoded_column = encoder.fit_transform(column)
            encoded_data.append(encoded_column)
        else:  # 直接添加数值列
            numeric_data.append(np.array([float(item) for item in column]))
    numeric_data = np.array(numeric_data)
    numeric_data = torch.transpose(torch.tensor(numeric_data), 0, 1)
    # 获取列表
    # encoded_matrix = list(np.hstack(encoded_data))
    encoded_data = np.array(encoded_data)
    combined_data = torch.transpose(torch.tensor(encoded_data), 0, 1)
    # position = len(encoded_data)  # 非float的位置

    data = torch.cat((combined_data, numeric_data), dim=1)
    data.requires_grad = True
    data.retain_graph = True

    if is_multi:
        return data.float().to(device), type_labels, None
    else:
        return data.float().to(device), anomaly_labels, None


def getids2017(args, is_multi):
    # 设定文件夹路径，例如 "./data/"
    folder_path = 'data/ids2017'

    # 使用glob.glob()匹配文件夹下的所有CSV文件
    # 这里假设CSV文件的扩展名是小写的".csv"，如果有大写的".CSV"，需要另外匹配或统一文件扩展名
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    # 按文件名排序，确保顺序读取
    csv_files.sort()

    print(csv_files)
    # 初始化一个空的DataFrame用于存储整合后的数据
    combined_data = []

    # 遍历文件列表，读取每个CSV文件，并将其追加到combined_df中
    train_data = []
    val_data = []
    test_data = []
    for idx, file in enumerate(csv_files):
        # 读取CSV文件
        df = pd.read_csv(file, encoding='ISO-8859-1')
        # df.dropna(inplace=True)
        df = df.dropna(subset=[df.columns[-1]])
        df = df[~df.isin([np.inf, -np.inf]).any(axis=1)]
        df.fillna(0, inplace=True)
        df.replace('-', -1, inplace=True)
        df.replace('', -2, inplace=True)
        df.replace(' ', -2, inplace=True)

        data_list = df.to_numpy()
        train, temp = train_test_split(data_list, test_size=(args.test_set + args.val_set),
                                       shuffle=False)
        val, test = train_test_split(data_list, test_size=(args.test_set / (args.test_set + args.val_set)),
                                     shuffle=False)
        if idx == 0:
            train_data = train
            val_data = val
            test_data = test
        else:
            train_data = np.vstack([train_data, train])
            val_data = np.vstack([val_data, val])
            test_data = np.vstack([test_data, test])

    combined_data = np.vstack([train_data, val_data, test_data])
    print(combined_data.shape)
    del data_list, train, temp, df, val, test, train_data, val_data, test_data
    gc.collect()
    # 创建LabelEncoder
    encoder = LabelEncoder()
    combined_data = np.delete(combined_data, 0, axis=1)  # 删除第一列
    combined_data[:, [1, 2]] = combined_data[:, [2, 1]]  # 把源ip和目的ip放到前两列
    if args.model_name == 'informer':
        time = convert_time_to_tensor(combined_data[:, 5])  # 获取时间列
    else:
        time = None
    combined_data = np.delete(combined_data, 5, axis=1)  # 删除时间列

    # 获取异常值
    type_labels = torch.tensor(encoder.fit_transform(combined_data[:, -1].astype(str))).long().to(device)  # 是哪种异常
    anomaly_labels = torch.where(type_labels != 0, 1, type_labels).long().to(device)  # 是否是异常(0, 1)
    combined_data = combined_data[:, :-1]  # 去除最后1列是最后的data

    print("Classes:", encoder.classes_)
    # 将ip转换成网段，添加到数据中，此时数据：【源IP地址，目的IP地址，源网段，目的网段，...】
    combined_data = ip_to_network_adjust_columns(combined_data)
    combined_data = list(map(list, zip(*combined_data)))
    # 分别对每列进行处理
    # 处理每一列（现在的每一行）
    transformed_columns = []
    gc.collect()
    for column in combined_data:  # data.shape[1]表示列的数量
        try:
            # 尝试将列转换为浮点数，如果成功，则假设它是数值型
            transformed_columns.append(list(np.array(column).astype(np.float32)))
        except ValueError:
            # 如果转换失败（抛出ValueError），则认为它是字符串型，应用LabelEncoder
            transformed_columns.append(encoder.fit_transform(column))
            # combined_data[:, i] = encoder.fit_transform(combined_data[:, i])
    del combined_data
    gc.collect()
    data = torch.tensor(transformed_columns).t().float()
    del transformed_columns
    gc.collect()
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = torch.tensor(data).to(device)

    data.requires_grad = True
    data.retain_graph = True

    if is_multi:
        return data.float().to(device), type_labels, time
    else:
        return data.float().to(device), anomaly_labels, time


def getids2019(args, is_multi):
    # 设定文件夹路径，例如 "./data/"
    folder_path = 'data/ids2019'

    # 使用glob.glob()匹配文件夹下的所有CSV文件
    # 这里假设CSV文件的扩展名是小写的".csv"，如果有大写的".CSV"，需要另外匹配或统一文件扩展名
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    # 按文件名排序，确保顺序读取
    csv_files.sort()

    print(csv_files)
    # 初始化一个空的DataFrame用于存储整合后的数据
    combined_data = []

    # 遍历文件列表，读取每个CSV文件，并将其追加到combined_df中
    train_data = []
    val_data = []
    test_data = []
    for idx, file in enumerate(csv_files):
        # 读取CSV文件
        df = pd.read_csv(file, encoding='ISO-8859-1')
        # df.dropna(inplace=True)
        df = df.dropna(subset=[df.columns[-1]])
        df = df[~df.isin([np.inf, -np.inf]).any(axis=1)]
        df.fillna(0, inplace=True)
        df.replace('-', -1, inplace=True)
        df.replace('', -2, inplace=True)
        df.replace(' ', -2, inplace=True)

        data_list = df.to_numpy()
        train, temp = train_test_split(data_list, test_size=(args.test_set + args.val_set),
                                       shuffle=False)
        val, test = train_test_split(data_list, test_size=(args.test_set / (args.test_set + args.val_set)),
                                     shuffle=False)
        if idx == 0:
            train_data = train
            val_data = val
            test_data = test
        else:
            train_data = np.vstack([train_data, train])
            val_data = np.vstack([val_data, val])
            test_data = np.vstack([test_data, test])
    combined_data = np.vstack([train_data, val_data, test_data])
    print(combined_data.shape)
    del data_list, train, temp, df, val, test, train_data, val_data, test_data
    gc.collect()
    # 创建LabelEncoder
    encoder = LabelEncoder()
    combined_data = np.delete(combined_data, 0, axis=1)  # 删除第一列
    combined_data = np.delete(combined_data, 0, axis=1)  # 删除第二列
    combined_data[:, [1, 2]] = combined_data[:, [2, 1]]  # 把源ip和目的ip放到前两列
    if args.model_name == 'informer':
        time = convert_time_to_tensor(combined_data[:, 5])  # 获取时间列
    else:
        time = None
    combined_data = np.delete(combined_data, 5, axis=1)  # 删除时间列

    # 获取异常值
    type_labels = torch.tensor(encoder.fit_transform(combined_data[:, -1].astype(str))).long().to(device)  # 是哪种异常
    anomaly_labels = torch.where(type_labels != 0, 1, type_labels).long().to(device)  # 是否是异常(0, 1)
    combined_data = combined_data[:, :-1]  # 去除最后1列是最后的data

    print("Classes:", encoder.classes_)
    # 将ip转换成网段，添加到数据中，此时数据：【源IP地址，目的IP地址，源网段，目的网段，...】
    combined_data = ip_to_network_adjust_columns(combined_data)
    combined_data = list(map(list, zip(*combined_data)))
    # 分别对每列进行处理
    # 处理每一列（现在的每一行）
    transformed_columns = []
    gc.collect()
    for idx, (column) in enumerate(combined_data):
        try:
            # 转换列为NumPy数组并立即转为列表，以减少内存占用
            temp = np.array(column, dtype=np.float32).tolist()
            # transformed_columns.append(temp_array)
        except ValueError:
            # 应用LabelEncoder，然后删除不再需要的局部变量
            temp = encoder.fit_transform(column).tolist()
        if idx == 0:
            transformed_columns = temp
        else:
            transformed_columns = np.vstack([transformed_columns, temp])
            # transformed_columns.append(temp_encoded)
        gc.collect()
    del combined_data
    gc.collect()
    data = torch.tensor(transformed_columns).t().float()
    del transformed_columns
    gc.collect()
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    data = torch.tensor(data).to(device)

    data.requires_grad = True
    data.retain_graph = True

    if is_multi:
        return data.float().to(device), type_labels, time
    else:
        return data.float().to(device), anomaly_labels, time


def read_data(args, is_multi=False):
    if args.data_type == 'unsw':
        return getUNSWData(args, is_multi)
    elif args.data_type == 'ids2017':
        return getids2017(args, is_multi)
    elif args.data_type == 'ids2019':
        return getids2019(args, is_multi)
    else:
        raise ValueError(f'Dataset {args.data_type} not available.')
