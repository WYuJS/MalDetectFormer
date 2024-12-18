import torch
import argparse
from utils.readData import read_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.mixture import GaussianMixture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    # 超参数
    parser.add_argument('--data_type', type=str, default='unsw', help='data type')
    parser.add_argument('--model_name', type=str, default='knn', help='model type')
    parser.add_argument('--is_multi', action='store_true', help='Binary classification(False) or multi classification(True)')
    parser.add_argument('--val_set', type=float, default = 0.1, help='validation set ratio')
    parser.add_argument('--test_set', type=float, default = 0.2, help='test set ratio')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    data, labels, position = read_data(args, args.is_multi)
    output_size = torch.unique(labels).numel()
    data = data.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=args.test_set, shuffle=False)
    model_name = args.model_name
    

    if model_name == 'gmm':
        gmm = GaussianMixture(n_components=output_size)
        # 训练模型
        gmm.fit(X_train, y_train)
        # 进行预测
        y_pred = gmm.predict(X_test)
        y_pred_proba = gmm.predict_proba(X_test)
    else:
        raise ValueError(f'Model {model_name} not available.')

    # 计算评估指标
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted" if args.is_multi else "binary", zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    

    print("Accuracy: {:.4f}".format(accuracy))
    print("F1 Score: {:.4f}".format(f1))
    print("Precision: {:.4f}".format(precision))
    print("Recall: {:.4f}".format(recall))
    auc = roc_auc_score(y_test, y_pred_proba if args.is_multi else y_pred_proba[:,1], multi_class='ovr' if args.is_multi else 'ovo')
    print("AUC: {:.4f}".format(auc))

    
