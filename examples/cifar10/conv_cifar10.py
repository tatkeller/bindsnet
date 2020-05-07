import torch
import argparse
import matplotlib.pyplot as plt
from torchvision import transforms
import os

from time import time as t
from tqdm import tqdm
import torch.nn as nn
from bindsnet.datasets import CIFAR10
from bindsnet.encoding import PoissonEncoder
from bindsnet.network import Network
from bindsnet.learning import PostPre
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import DiehlAndCookNodes, Input
from bindsnet.network.topology import Conv2dConnection, Connection
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_conv2d_weights,
    plot_voltages,
)

print()

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--n_test", type=int, default=400)
parser.add_argument("--kernel_size", type=int, default=16)
parser.add_argument("--stride", type=int, default=4)
parser.add_argument("--n_filters", type=int, default=25)
parser.add_argument("--padding", type=int, default=0)
parser.add_argument("--time", type=int, default=50)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128.0)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=False, gpu=False, train=True)
parser.add_argument("--n_train", type=int, default=500)

args = parser.parse_args()

seed = args.seed
n_epochs = args.n_epochs
n_test = args.n_test
kernel_size = args.kernel_size
stride = args.stride
n_filters = args.n_filters
padding = args.padding
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu
n_train = args.n_train

if gpu:
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)

if not train:
    update_interval = n_test

conv_size = int((32 - kernel_size + 2 * padding) / stride) + 1
per_class = int((n_filters * conv_size * conv_size) / 10)

# Build network.
network = Network()
input_layer = Input(n=3072, shape=(3, 32, 32), traces=True)

conv_layer = DiehlAndCookNodes(
    n=n_filters * conv_size * conv_size,
    shape=(n_filters, conv_size, conv_size),
    traces=True,
)

conv_conn = Conv2dConnection(
    input_layer,
    conv_layer,
    kernel_size=kernel_size,
    stride=stride,
    update_rule=PostPre,
    norm=0.4 * kernel_size ** 2,
    nu=[1e-4, 1e-2],
    wmax=1.0,
)

w = torch.zeros(n_filters, conv_size, conv_size, n_filters, conv_size, conv_size)
for fltr1 in range(n_filters):
    for fltr2 in range(n_filters):
        if fltr1 != fltr2:
            for i in range(conv_size):
                for j in range(conv_size):
                    w[fltr1, i, j, fltr2, i, j] = -100.0

w = w.view(n_filters * conv_size * conv_size, n_filters * conv_size * conv_size)
recurrent_conn = Connection(conv_layer, conv_layer, w=w)

network.add_layer(input_layer, name="X")
network.add_layer(conv_layer, name="Y")
network.add_connection(conv_conn, source="X", target="Y")
network.add_connection(recurrent_conn, source="Y", target="Y")

# Voltage recording for excitatory and inhibitory layers.
voltage_monitor = Monitor(network.layers["Y"], ["v"], time=time)
network.add_monitor(voltage_monitor, name="output_voltage")

if gpu:
    network.to("cuda")

# Load CIFAR10 data.
train_dataset = CIFAR10(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "CIFAR10"),
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)
    
test_dataset = CIFAR10(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "CIFAR10"),
    train=False,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), 
         #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         transforms.Lambda(lambda x: x * intensity)] 
    ),
)

spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(network.layers[layer], state_vars=["v"], time=time)
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

# Train the network.
print("Begin training.\n")
start = t()
print("OHIOYOYOYOYOYOYOYO")
inpt_axes = None
inpt_ims = None
spike_ims = None
spike_axes = None
weights1_im = None
voltage_ims = None
voltage_axes = None
test_pairs = []
training_pairs = []


for epoch in range(n_epochs):
    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=gpu
    )

    for step, batch in enumerate(tqdm(train_dataloader)):
        # Get next input sample.

        if step > n_train:
            break

        inputs = {"X": batch["encoded_image"].view(time, 1, 3, 32, 32)}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        label = batch["label"]

        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1)
        training_pairs.append([spikes["Y"].get("s").sum(0), label])

        # Optionally plot various simulation information.
        if plot:
            image = batch["image"].view(32, 32)

            inpt = inputs["X"].view(time, 3072).sum(0).view(32, 32)
            weights1 = conv_conn.w
            _spikes = {
                "X": spikes["X"].get("s").view(time, -1),
                "Y": spikes["Y"].get("s").view(time, -1),
            }
            _voltages = {"Y": voltages["Y"].get("v").view(time, -1)}

            inpt_axes, inpt_ims = plot_input(
                image, inpt, label=label, axes=inpt_axes, ims=inpt_ims
            )
            spike_ims, spike_axes = plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
            weights1_im = plot_conv2d_weights(weights1, im=weights1_im)
            voltage_ims, voltage_axes = plot_voltages(
                _voltages, ims=voltage_ims, axes=voltage_axes
            )

            plt.pause(1)

        network.reset_state_variables()  # Reset state variables.

print("Progress: %d / %d (%.4f seconds)\n" % (n_train, n_train, t() - start))
print("Training complete.\n")

# Define logistic regression model using PyTorch.
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        # h = int(input_size/2)
        self.linear_1 = nn.Linear(input_size, num_classes)
        # self.linear_1 = nn.Linear(input_size, h)
        # self.linear_2 = nn.Linear(h, num_classes)

    def forward(self, x):
        out = torch.sigmoid(self.linear_1(x.float().view(-1)))
        # out = torch.sigmoid(self.linear_2(out))
        return out

n_neurons = 625
examples = 500
n_epochs = 300
gpu = False
num_classes = 10
cifar10_input= 3 * 32 * 32
cifar10_shape = (3, 32, 32)

# Create and train logistic regression model on reservoir outputs.
model = NN(n_neurons, num_classes) #.to(device_id)
criterion = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

# Training the Model
print("\n Training the read out")
pbar = tqdm(enumerate(range(n_epochs)))
for epoch, _ in pbar:
    avg_loss = 0
    for i, (s, l) in enumerate(training_pairs):
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(s)
        label = torch.zeros(1, 1, 10).float() #.to(device_id)
        label[0, 0, l] = 1.0
        loss = criterion(outputs.view(1, 1, -1), label)
        avg_loss += loss.data
        loss.backward()
        optimizer.step()

    pbar.set_description_str(
        "Epoch: %d/%d, Loss: %.4f"
        % (epoch + 1, n_epochs, avg_loss / len(training_pairs))
    )

n_iters = examples
test_pairs = []
test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=gpu
    )
pbar = tqdm(enumerate(test_dataloader))
network.train(False)
for step, batch in enumerate(tqdm(test_dataloader)):
    # Get next input sample.

    if step > n_test:
        break

    inputs = {"X": batch["encoded_image"].view(time, 1, 3, 32, 32)}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    label = batch["label"]

    # Run the network on the input.
    network.run(inputs=inputs, time=time, input_time_dim=1)
    test_pairs.append([spikes["Y"].get("s").sum(0), label])

    # Optionally plot various simulation information.
    if plot:
        image = batch["image"].view(32, 32)

        inpt = inputs["X"].view(time, 3072).sum(0).view(32, 32)
        weights1 = conv_conn.w
        _spikes = {
            "X": spikes["X"].get("s").view(time, -1),
            "Y": spikes["Y"].get("s").view(time, -1),
        }
        _voltages = {"Y": voltages["Y"].get("v").view(time, -1)}

        inpt_axes, inpt_ims = plot_input(
            image, inpt, label=label, axes=inpt_axes, ims=inpt_ims
        )
        spike_ims, spike_axes = plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
        weights1_im = plot_conv2d_weights(weights1, im=weights1_im)
        voltage_ims, voltage_axes = plot_voltages(
            _voltages, ims=voltage_ims, axes=voltage_axes
        )

        plt.pause(1)

    network.reset_state_variables()  # Reset state variables.

# Test the Model
correct, total = 0, 0
model.eval()
for s, label in test_pairs:
    outputs = model(s)
    _, predicted = torch.max(outputs.data.unsqueeze(0), 1)
    total += 1
    correct += int(predicted == label.long()) #.to(device_id))

print(
    "\n Accuracy of the model on %d test images: %.2f %%"
    % (n_iters, 100 * correct / total)
)