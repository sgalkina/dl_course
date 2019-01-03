import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.utils.data

from sklearn.model_selection import train_test_split

print('GPU device count', torch.cuda.device_count())
device = torch.device("cuda:0")
print("Device in use:", device)

shifts = list(range(150, 721, 30))
merged = np.zeros((2286, len(shifts), 2, 25000))
for c, i in enumerate(shifts):
    print(c)
    merged[:, c, :, :] = np.load('../dataset_shifted_unique/{}.npy'.format(i))

# merged = np.load('merged_shifted_30.npy')
labels = np.load('../data/df_unique_labels.npy')
X_dataset_unique = np.load('../data/dataset_unique.npy')

X_train_full, X_test_full, X_train_shifted, X_test_shifted, y_train, y_test = train_test_split(
    X_dataset_unique,
    merged,
    labels,
    test_size=0.2,
    random_state=1
)

print(X_train_full.shape, X_train_shifted.shape, y_train.shape)
print(X_test_full.shape, X_train_shifted.shape, y_test.shape)


class SpectralDatasetShifted(torch.utils.data.dataset.Dataset):
    """Spectra dataset"""

    def __init__(self, X, X_s, y):
        self.X = X
        self.X_s = X_s
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.X_s[idx], self.y[idx]


trainset = SpectralDatasetShifted(X_train_full, X_train_shifted, y_train)
testset = SpectralDatasetShifted(X_test_full, X_test_shifted, y_test)

batch_size = 16

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

import torch.nn.init as init


# data (256, 2, 50000, 200) --- (batch_size, 2, len_spectrum, n_shifts)
# kernel (2, 10, 200) --- ((kernel_size), n_shifts)

# define the simplest network
class Net(nn.Module):

    def __init__(self, num_output):
        super(Net, self).__init__()

        # linear fully connected layers
        self.W_1_full = Parameter(init.kaiming_normal_(torch.Tensor(512, 25000)))
        self.b_1_full = Parameter(init.constant_(torch.Tensor(512), 0))

        self.W_1 = Parameter(init.kaiming_normal_(torch.Tensor(512, 199680)))
        self.b_1 = Parameter(init.constant_(torch.Tensor(512), 0))

        self.W_2 = Parameter(init.kaiming_normal_(torch.Tensor(256, 512 + 512)))
        self.b_2 = Parameter(init.constant_(torch.Tensor(256), 0))

        self.W_3 = Parameter(init.kaiming_normal_(torch.Tensor(128, 256)))
        self.b_3 = Parameter(init.constant_(torch.Tensor(128), 0))

        self.W_4 = Parameter(init.kaiming_normal_(torch.Tensor(64, 128)))
        self.b_4 = Parameter(init.constant_(torch.Tensor(64), 0))

        self.W_5 = Parameter(init.kaiming_normal_(torch.Tensor(num_output, 64)))
        self.b_5 = Parameter(init.constant_(torch.Tensor(num_output), 0))

        self.activation = torch.nn.ReLU()

        self.dropout = nn.Dropout2d(p=0.5)

        # LSTM layer to utilize the sequential nature of the data
        #         self.rnn = nn.LSTM(1, self.hidden_size, batch_first=True)

        # CNN to find masses shifts
        self.conv1 = nn.Conv2d(in_channels=len(shifts), out_channels=64, kernel_size=(2, 10))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 5))

    def forward(self, x_full, x):
        x = x.float().to(device)
        x_full = x_full.float().to(device)

        x = F.max_pool2d(F.relu(self.conv1(x)), (1, 6), stride=(1, 4))
        #         print(x.shape)
        x = F.max_pool2d(F.relu(self.conv2(x)), (1, 6), stride=(1, 4))
        #         print(x.shape)
        x = x.view(-1, self.num_flat_features(x))
        x = F.linear(x, self.W_1, self.b_1)
        x = self.activation(x)

        x_full = self.dropout(x_full)
        x_full = F.linear(x_full, self.W_1_full, self.b_1_full)
        x_full = self.activation(x_full)

        x = F.linear(torch.cat([x, x_full], 1), self.W_2, self.b_2)
        x = self.activation(x)
        x = F.linear(x, self.W_3, self.b_3)
        x = self.activation(x)
        x = F.linear(x, self.W_4, self.b_4)
        x = self.activation(x)
        return F.linear(x, self.W_5, self.b_5)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


n_categories = y_train.shape[1]
net = Net(n_categories).to(device)
optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-2)
criterion = nn.MultiLabelSoftMarginLoss()


def train(spectrum_tensor, spectrum_tensor_shifted, category_tensor):
    spectrum_tensor.to(device)
    category_tensor.to(device)
    net.train()
    output = net(Variable(spectrum_tensor).to(device), Variable(spectrum_tensor_shifted).to(device))
    batch_loss = criterion(output, Variable(category_tensor).float().to(device))
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()
    return output, batch_loss


def evaluate(spectrum_tensor, spectrum_tensor_shifted):
    spectrum_tensor.to(device)
    net.eval()
    output = net(Variable(spectrum_tensor).to(device), Variable(spectrum_tensor_shifted).to(device))
    return output


from sklearn.metrics import hamming_loss, f1_score, recall_score, precision_score


num_epochs = 100
epoch_train_loss = []

av_train_accuracies = []
av_val_accuracies = []
filepath = 'output.txt'

with open('test_gpu.txt', 'w') as f:
    f.write(str(list(net.parameters())[0]))


train_acc, valid_acc = [], []


with open(filepath, 'w') as f:
    f.write('epoch,epoch_loss,train_loss,valid_loss,hamm_train_loss,hamm_valid_loss\n')
for epoch in range(num_epochs):
    print(epoch)

    # Train the net
    loss = 0
    c = 0
    for x, x_s, y in train_loader:
        c += 1
        output, batch_loss = train(x, x_s, y)
        loss += float(batch_loss)
    print(c, 'batches')
    epoch_train_loss.append(loss / c)

    # Evaluate accuracy on the train dataset
    bloss = 0
    for x, x_s, y in train_loader:
        preds = evaluate(x, x_s)
        predicted_labels = torch.sigmoid(preds).data.cpu().numpy() > 0.5
        loss, count = 0, 0
        for i in range(x.shape[0]):
            count += 1
            loss += hamming_loss(y.data.cpu().numpy()[i], predicted_labels[i])
        av_train_accuracies.append(loss / count)
        bloss += float(criterion(preds, Variable(y).float().to(device)))
    train_acc.append(bloss / batch_size)

    # Evaluate accuracy on the test dataset

    predictions = np.zeros(y_test.shape)
    cur = 0
    bloss = 0
    for x, x_s, y in test_loader:
        preds = evaluate(x, x_s)
        val_preds = list(preds.data.cpu().numpy())
        val_targs = list(y.data.cpu().numpy())
        predicted_labels = torch.sigmoid(preds).data.cpu().numpy() > 0.5
        loss, count = 0, 0
        for i in range(y.shape[0]):
            count += 1
            loss += hamming_loss(y.data.cpu().numpy()[i], predicted_labels[i])
            predictions[cur + i, :] = predicted_labels[i]
        cur += batch_size
        bloss += float(criterion(preds, Variable(y).float().to(device)))
        av_val_accuracies.append(loss / count)
    valid_acc.append(bloss / batch_size)

    f_scores = f1_score(y_true=y_test, y_pred=predictions, average=None)
    precision_scores = precision_score(y_true=y_test, y_pred=predictions, average=None)
    recall_scores = recall_score(y_true=y_test, y_pred=predictions, average=None)

    np.save('predictions_{}.npy'.format(epoch), predictions)
    np.save('f_scores_{}.npy'.format(epoch), f_scores)
    np.save('precision_scores_{}.npy'.format(epoch), precision_scores)
    np.save('recall_scores_{}.npy'.format(epoch), recall_scores)
    with open(filepath, 'a') as f:
        f.write('{},{},{},{},{},{}\n'.format(
            epoch + 1,
            epoch_train_loss[-1],
            train_acc[-1],
            valid_acc[-1],
            av_train_accuracies[-1],
            av_val_accuracies[-1]
        ))
