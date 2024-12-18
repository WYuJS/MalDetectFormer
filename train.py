import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model.transformer import TransformerModel
from model.CNN import CNN
from model.Autoformer import Autoformer
from model.Informer import Informer
from model.FEDformer import FEDformer
from model.LSTM import LSTM
import argparse
from model.BiLSTMCNN import BiLSTMCNN
from model.MalDetectFormer import MalDetectFormer
from torch.utils.data import TensorDataset, DataLoader
from utils.readData import read_data
from utils.load_dataloader import load_dataloader
from model.EGCN import getGraph
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 输入参数
def parse_args():
    parser = argparse.ArgumentParser()
    # 超参数
    parser.add_argument('--data_type', type=str, default='unsw', help='data type')
    parser.add_argument('--model_name', type=str, default='maldetectformer', help='model type')
    parser.add_argument('--is_multi', action='store_true',
                        help='Binary classification(False) or multi classification(True)')
    parser.add_argument('--num_epochs', type=int, default=100, help='train model eopchs')
    parser.add_argument('--batch_size', type=int, default=256, help='The size of each training input')
    parser.add_argument('--egcn_hidden_size', type=int, default=16, help='The size of egcn hidden')
    parser.add_argument('--lr', type=float, default=0.0003, help='The learning rate of the overall model')
    parser.add_argument('--val_set', type=float, default=0.1, help='validation set ratio')
    parser.add_argument('--test_set', type=float, default=0.2, help='test set ratio')
    parser.add_argument('--patience', type=int, default=1, help='Verification set stop count')

    # 模型的参数
    parser.add_argument('--d_model', type=int, default=512,
                        help='Embedding dimension of word vectors in the NetSentinelformer model')
    parser.add_argument('--nhead', type=int, default=8,
                        help='The number of heads in the attention mechanism of the form model')
    parser.add_argument('--num_encoder_layers', type=int, default=3, help='The number of encoder')
    parser.add_argument('--dim_feedforward', type=int, default=2048,
                        help='The dimension of transformation matrix in feedforward fully connected networks')
    parser.add_argument('--num_decoder_layers', type=int, default=3, help='The number of decoder')
    parser.add_argument('--num_intervals', type=int, default=64, help='The number of data within the interval')
    parser.add_argument('--attn_dropout', type=float, default=0.2, help='Init dropout rate of dot attention mechanism')
    parser.add_argument('--all_dropout', type=float, default=0.3, help='Dropout rate of all modules')
    parser.add_argument('--hidden_feats', type=int, default=16, help='The hidden layer size of egcn')
    parser.add_argument('--alpha', type=float, default=0.1, help='The first dropout adjustment rate')
    parser.add_argument('--beta', type=float, default=0.1, help='The second dropout adjustment rate')

    return parser.parse_args()


def compute_dropout_rate(current_loss, last_loss, args):
    p = args.attn_dropout + args.alpha * np.tanh(args.beta * (current_loss - last_loss))
    last_loss = (current_loss + last_loss) / 2
    if p < 0:
        p = 0
    return p, last_loss


def get_model(args, egcn=None):
    if args.model_name == 'maldetectformer':
        model = MalDetectFormer(source_vocab=args.num_features,
                                target_vocab=args.num_features,
                                output_size=args.output_size,
                                num_encoder_layers=args.num_encoder_layers,
                                num_decoder_layers=args.num_decoder_layers,
                                d_model=args.d_model,
                                d_ff=args.dim_feedforward,
                                head=args.nhead,
                                dropout=args.all_dropout,
                                egcn_hidden_size=args.egcn_hidden_size,  # EGCN隐藏层
                                g=egcn['g'],  # ip图
                                G=egcn['G'],  # net图
                                g_node_name_to_index=egcn['ip_node_name_to_index'],  # g的节点映射表
                                G_node_name_to_index=egcn['net_node_name_to_index'],  # G的节点映射表
                                )
    elif args.model_name == 'transformer':
        model = TransformerModel(args.num_features, args.output_size).to(device)
    elif args.model_name == 'cnn':
        model = CNN(args.num_features, args.output_size)
    elif args.model_name == 'lstm':
        model = LSTM(args.num_features, args.output_size)
    elif args.model_name == 'bilstmcnn':
        model = BiLSTMCNN(args.num_features, args.output_size)
    elif args.model_name == 'informer':
        model = Informer(enc_in=args.num_features, dec_in=args.num_features, c_out=args.output_size)
    elif args.model_name == 'fedformer':
        model = FEDformer(enc_in=args.num_features, dec_in=args.num_features, c_out=args.num_features,
                          output_size=args.output_size)
    elif args.model_name == 'autoformer':
        model = Autoformer(enc_in=args.num_features, dec_in=args.num_features, c_out=args.num_features,
                           output_size=args.output_size)
    else:
        raise ValueError(f'Model {args.model_name} not available.')

    return model


def model_predict(model, data, tmbed, p, args):
    if args.model_name == 'maldetectformer':
        return model(data, data, args.num_intervals, p, args.is_multi)
    elif args.model_name == 'transformer':
        return model(data, data, args.is_multi)
    elif args.model_name == 'cnn':
        return model(data, args.is_multi)
    elif args.model_name == 'lstm':
        return model(data, args.is_multi)
    elif args.model_name == 'bilstmcnn':
        return model(data, args.is_multi)
    elif args.model_name == 'informer':
        return model(data, tmbed, data, tmbed, args.is_multi)
    elif args.model_name == 'fedformer':
        return model(data, tmbed, data, tmbed, args.is_multi)
    elif args.model_name == 'autoformer':
        return model(data, tmbed, data, tmbed, args.is_multi)


# 训练模型
def train(args):
    # 读取数据
    data, labels, tmbed = read_data(args, args.is_multi)
    args.num_features = data.shape[1]  # 特征数量
    args.output_size = torch.unique(labels).numel()  # 输出的维度

    if args.model_name == 'maldetectformer':
        egcn = getGraph(data)
    else:
        egcn = None

    model = get_model(args, egcn)

    if args.is_multi:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 获取DataLoader
    train_dataloader, val_dataloader, test_dataloader = load_dataloader(data, labels, tmbed, args)

    # 训练模型
    early_stopping_counter = 0
    best_val_loss = float('inf')
    last_loss = 0
    p = args.attn_dropout
    best_model_name = 'result/' + args.model_name + '_' + args.data_type + '_' + str(
        args.is_multi) + '_' + 'best_model.pth'
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        for idx, (data_batch, tmbed_batch, labels_batch) in enumerate(train_dataloader):
            torch.cuda.empty_cache()
            output = model_predict(model, data_batch, tmbed_batch, p, args)
            _, predicted_classes = torch.max(output, dim=1)
            if args.is_multi is False:
                output = output[:, 1]
                labels_batch = labels_batch.float()

            output = output.float()
            loss = criterion(output, labels_batch)

            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            if idx % 100 == 0:
                print(f"  Epoch {epoch + 1}   idx:{idx}, Loss: {loss:.4f}")
        print(f"My model:{args.model_name}  Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader):.4f}")

        # 验证模型性能
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data_batch, tmbed_batch, labels_batch in val_dataloader:
                output = model_predict(model, data_batch, tmbed_batch, p, args)
                _, predicted_classes = torch.max(output, dim=1)
                if args.is_multi is False:
                    output = output[:, 1]
                    labels_batch = labels_batch.float()
                output = output.float()
                loss = criterion(output, labels_batch)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        print(f"Epoch {epoch + 1},VAL Loss: {val_loss:.4f}")
        p, last_loss = compute_dropout_rate(val_loss, last_loss, args)
        # 检查是否有改善
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0  # 重置早停计数器
            # 可以在这里保存模型
            torch.save(model.state_dict(), best_model_name)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(f'Early stopping triggered after epoch {epoch + 1}.')
                break  # 退出训练循环

    # 测试模型性能
    model.load_state_dict(torch.load(best_model_name))
    model.eval()
    test_loss = 0
    true_labels = []
    predictions = []
    output_result = []
    probs = []
    with torch.no_grad():
        for inputs, tmbed_batch, labels_batch in test_dataloader:
            output = model_predict(model, inputs, tmbed_batch, 0, args)
            if args.is_multi:
                output = F.softmax(output)
            else:
                output = F.sigmoid(output)
            _, predicted_classes = torch.max(output, dim=1)
            if args.is_multi is False:
                output = output[:, 1]
                labels_batch = labels_batch.float()
            output = output.float()

            loss = criterion(output, labels_batch)
            test_loss += loss.item()
            output_result.extend(output.tolist())
            true_labels.extend(labels_batch.tolist())
            predictions.extend(predicted_classes.tolist())
        print(f"Result Loss: {test_loss / len(test_dataloader):.4f}")

    # 计算各种指标
    output_result = np.array(output_result)
    accuracy = accuracy_score(true_labels, predictions)
    print(f'Accuracy: {accuracy}')
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average="weighted",
                                                               zero_division=0)
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    if args.is_multi:
        auc = roc_auc_score(true_labels, output_result, multi_class='ovr')
    else:
        auc = roc_auc_score(true_labels, output_result)
    print(f'AUC: {auc:.4f}')


if __name__ == '__main__':
    torch.cuda.empty_cache()
    args = parse_args()
    train(args)