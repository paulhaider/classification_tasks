#!/usr/bin/env python3
# Example usage of the Yin-Yang dataset

import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import ClassificationTask
from torch.utils.data import DataLoader

# Setup datasets (training, validation and test set)
dataset_train = ClassificationTask(size=5000, seed=42)
dataset_validation = ClassificationTask(size=1000, seed=41)
dataset_test = ClassificationTask(size=1000, seed=40)

# Setup datasets (training, validation and test set)
batchsize_train = 20
batchsize_eval = len(dataset_test)

train_loader = DataLoader(dataset_train, batch_size=batchsize_train, shuffle=True)
val_loader = DataLoader(dataset_validation, batch_size=batchsize_eval, shuffle=True)
test_loader = DataLoader(dataset_test, batch_size=batchsize_eval, shuffle=False)

# Plot data
def plot_data():
    fig, axes = plt.subplots(ncols=3, sharey=True, figsize=(15, 8))
    titles = ['Training set', 'Validation set', 'Test set']
    for i, loader in enumerate([train_loader, val_loader, test_loader]):
        axes[i].set_title(titles[i])
        axes[i].set_aspect('equal', adjustable='box')
        xs = []
        ys = []
        cs = []
        for batch, batch_labels in loader:
            for j, item in enumerate(batch):
                x1, y1, x2, y2 = item
                c = int(np.where(batch_labels[j] == 1)[0])
                xs.append(x1)
                ys.append(y1)
                cs.append(c)
        xs = np.array(xs)
        ys = np.array(ys)
        cs = np.array(cs)
        axes[i].scatter(xs[cs == 0], ys[cs == 0], color='C0', edgecolor='k', alpha=0.7)
        axes[i].scatter(xs[cs == 1], ys[cs == 1], color='C1', edgecolor='k', alpha=0.7)
        axes[i].scatter(xs[cs == 2], ys[cs == 2], color='C2', edgecolor='k', alpha=0.7)
        axes[i].set_xlabel('x1')
        if i == 0:
            axes[i].set_ylabel('y1')
        fig.savefig('./data.pdf')

# Setup ANN
class Net(torch.nn.Module):
    def __init__(self, network_layout, noise_width=0):
        super(Net, self).__init__()
        self.noise_width = noise_width
        self.n_inputs = network_layout['n_inputs']
        self.n_layers = network_layout['n_layers']
        self.layer_sizes = network_layout['layer_sizes']
        self.layers = torch.nn.ModuleList()
        layer = torch.nn.Linear(self.n_inputs, self.layer_sizes[0], bias=True)
        self.layers.append(layer)
        for i in range(self.n_layers-1):
            layer = torch.nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1], bias=True)
            self.layers.append(layer)
        return

    def forward(self, x):
        x_hidden = []
        for i in range(self.n_layers):
            x = self.layers[i](x)
            if not i == (self.n_layers-1):
                relu = torch.nn.ReLU()
                if self.noise_width != 0:
                    x = relu(x)
                    x += torch.normal(mean=torch.zeros_like(x), std=self.noise_width)
                else:
                    x = relu(x)
                x_hidden.append(x)
        return x

# Linear classifier for reference
shallow_network_layout = {
    'n_inputs': 4,
    'n_layers': 1,
    'layer_sizes': [3],
}
linear_classifier = Net(shallow_network_layout)

# used to determine validation accuracy after each epoch in training
def validation_step(net, criterion, loader):
    with torch.no_grad():
        num_correct = 0
        num_shown = 0
        for j, data in enumerate(loader):
            inputs, labels = data
            # need to convert to float32 because data is in float64
            inputs = inputs.float()
            outputs = net(inputs)
            winner = outputs.argmax(1)
            num_correct += len(outputs[winner == labels.argmax(1)])
            num_shown += len(labels)
        accuracy = float(num_correct) / num_shown
    return accuracy


def train(noise_width=0):

    torch.manual_seed(12345)
    # ANN with one hidden layer (with 120 neurons)
    network_layout = {
        'n_inputs': 4,
        'n_layers': 2,
        'layer_sizes': [150, 3],
       }
    net = Net(network_layout, noise_width)

    # set training parameters
    n_epochs = 500
    learning_rate = 0.1
    val_accuracies = []
    train_accuracies = []
    # setup loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    # train for n_epochs
    for epoch in range(n_epochs):
        val_acc = validation_step(net, criterion, val_loader)
        if epoch % 25 == 0:
            print('Validation accuracy after {0} epochs: {1}'.format(epoch, val_acc))
        val_accuracies.append(val_acc)
        num_correct = 0
        num_shown = 0
        for j, data in enumerate(train_loader):
            inputs, labels = data
            # need to convert to float32 because data is in float64
            inputs = inputs.float()
            labels = labels.float()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass
            outputs = net(inputs)
            winner = outputs.argmax(1)
            num_correct += len(outputs[outputs.argmax(1) == labels.argmax(1)])
            num_shown += len(labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        accuracy = float(num_correct) / num_shown
        train_accuracies.append(accuracy)

    # after training evaluate on test set
    test_acc = validation_step(net, criterion, test_loader)
    print('Final test accuracy:', test_acc)
    return test_acc, train_accuracies

if __name__ == "__main__":

    widths = [0., 0.01, 0.05, 0.1, 0.2, 0.4, 0.8]
    test_accs = []
    train_accs = []
    for width in widths:
        test_acc, train_acc = train(width)
        test_accs.append(test_acc)
        train_accs.append(train_acc)

    test_accs = np.array(test_accs)
    train_accs = np.array(train_accs)

    fig, axes = plt.subplots(nrows=2, figsize=(8,8))
    fig.suptitle('Yin-Yang training with noisy ANN')
    for i, accs in enumerate(train_accs):
        epoch_x = np.arange(len(accs))
        axes[0].plot(epoch_x, accs, label=widths[i], linewidth=0.8)
    axes[1].plot(widths, test_accs, 'x-', linewidth=0.8)
    axes[0].set_xlabel('epochs')
    axes[0].set_ylabel('train accuracy')
    axes[0].legend(title=r'$\sigma_\mathrm{noise}$')
    axes[0].grid(True)
    axes[1].set_xlabel(r'$\sigma_\mathrm{noise}$')
    axes[1].set_ylabel('test accuracy')
    axes[1].grid(True)
    plt.tight_layout()
    # plt.show()
    for form in ['pdf', 'svg']:
        plt.savefig('./noisy_yinyang.' + form)
