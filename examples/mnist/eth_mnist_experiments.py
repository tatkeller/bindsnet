import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from torchvision import transforms
from tqdm import tqdm

from time import time as t

from bindsnet.datasets import MNIST, CIFAR10
from bindsnet.encoding import PoissonEncoder
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_weights,
    plot_assignments,
    plot_performance,
    plot_voltages,
)


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--logistic_epochs", type=int, default=100)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_train", type=int, default=100)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=False, gpu=False, train=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_epochs = args.n_epochs
logistic_epochs = args.logistic_epochs
n_test = args.n_test
n_train = args.n_train
n_workers = args.n_workers
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu

# Sets up Gpu use
# if gpu:
#     torch.cuda.manual_seed_all(seed)
# else:
#     torch.manual_seed(seed)

# Determines number of workers to use
if n_workers == -1:
    n_workers = gpu * 4 * torch.cuda.device_count()

if not train:
    update_interval = n_test

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity

device_id = 'cpu'
if gpu and torch.cuda.is_available():
    device_id = 'gpu'

# Build network.
network = DiehlAndCook2015(
    n_inpt=784,
#    n_inpt=3072,
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    dt=dt,
    norm=78.4,
    theta_plus=theta_plus,
    inpt_shape=(1, 28, 28)
#    inpt_shape=(3, 32, 32),
)

# Directs network to GPU
if gpu:
    network.to("cuda")

# Load MNIST data.
dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

#print(dataset)

dataset_test = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST_TEST"),
    train=False,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)
#print(dataset_test)


## Load CIFAR-10 data.
#dataset = CIFAR10(
#    PoissonEncoder(time=time, dt=dt),
#    None,
#    root=os.path.join("..", "..", "data", "CIFAR10"),
#    train=True,
#    download=True,
#    transform=transforms.Compose(
#        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
#    ),
#)
#
#dataset_test = CIFAR10(
#    PoissonEncoder(time=time, dt=dt),
#    None,
#    root=os.path.join("..", "..", "data", "CIFAR10_TEST"),
#    train=False,
#    download=True,
#    transform=transforms.Compose(
#        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
#    ),
#)

# Record spikes during the simulation.
spike_record = torch.zeros(update_interval, time, n_neurons)

# Neuron assignments and spike proportions.
n_classes = 10
assignments = -torch.ones(n_neurons)
proportions = torch.zeros(n_neurons, n_classes)
rates = torch.zeros(n_neurons, n_classes)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(network.layers["Ae"], ["v"], time=time)
inh_voltage_monitor = Monitor(network.layers["Ai"], ["v"], time=time)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")

# Set up monitors for spikes and voltages
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(network.layers[layer], state_vars=["v"], time=time)
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None

# Train the network.
print("\nBegin training.\n")
start = t()


training_pairs = []
for epoch in range(n_epochs):
    labels = []

    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    # Create a dataloader to iterate and batch data
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=n_workers, pin_memory=gpu
    )

    for step, batch in enumerate(tqdm(dataloader)):

        if step > n_train:
            break
        # Get next input sample.
        # print(batch['encoded_image'].shape)

        inputs = {"X": batch["encoded_image"].view(time, 1, 1, 28, 28)}
#        inputs = {"X": batch["encoded_image"].view(time, 1, 3, 32, 32)}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        if step % update_interval == 0 and step > 0:
            # Convert the array of labels into a tensor
            label_tensor = torch.tensor(labels)

            # Get network predictions.
            all_activity_pred = all_activity(
                spikes=spike_record, assignments=assignments, n_labels=n_classes
            )
            proportion_pred = proportion_weighting(
                spikes=spike_record,
                assignments=assignments,
                proportions=proportions,
                n_labels=n_classes,
            )

            # Compute network accuracy according to available classification strategies.
            accuracy["all"].append(
                100
                * torch.sum(label_tensor.long() == all_activity_pred).item()
                / len(label_tensor)
            )
            accuracy["proportion"].append(
                100
                * torch.sum(label_tensor.long() == proportion_pred).item()
                / len(label_tensor)
            )

            print(
                "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                % (
                    accuracy["all"][-1],
                    np.mean(accuracy["all"]),
                    np.max(accuracy["all"]),
                )
            )
            print(
                "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f (best)\n"
                % (
                    accuracy["proportion"][-1],
                    np.mean(accuracy["proportion"]),
                    np.max(accuracy["proportion"]),
                )
            )

            # Assign labels to excitatory layer neurons.
            assignments, proportions, rates = assign_labels(
                spikes=spike_record,
                labels=label_tensor,
                n_labels=n_classes,
                rates=rates,
            )

            labels = []

        labels.append(batch["label"])

        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1)
        if epoch == n_epochs - 1:
            training_pairs.append([spikes["Ae"].get("s").sum(0), batch['label']])

        # Get voltage recording.
        exc_voltages = exc_voltage_monitor.get("v")
        inh_voltages = inh_voltage_monitor.get("v")

        # Add to spikes recording.
        spike_record[step % update_interval] = spikes["Ae"].get("s").squeeze()

        # Optionally plot various simulation information.
        if plot:
            image = batch["image"].view(28, 28)
            inpt = inputs["X"].view(time, 784).sum(0).view(28, 28)
            input_exc_weights = network.connections[("X", "Ae")].w
            square_weights = get_square_weights(
                input_exc_weights.view(784, n_neurons), n_sqrt, 28
            )
            square_assignments = get_square_assignments(assignments, n_sqrt)
            spikes_ = {layer: spikes[layer].get("s") for layer in spikes}
            voltages = {"Ae": exc_voltages, "Ai": inh_voltages}
            inpt_axes, inpt_ims = plot_input(
                image, inpt, label=batch["label"], axes=inpt_axes, ims=inpt_ims
            )
            spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
            weights_im = plot_weights(square_weights, im=weights_im)
            assigns_im = plot_assignments(square_assignments, im=assigns_im)
            perf_ax = plot_performance(accuracy, ax=perf_ax)
            voltage_ims, voltage_axes = plot_voltages(
                voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
            )

            plt.pause(1e-8)

        network.reset_state_variables()  # Reset state variables.

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Training complete.\n")



############################
# Logistic Regression Model
############################

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


# Create and train logistic regression model on reservoir outputs.
model = NN(n_neurons, 10).to(device_id)
criterion = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

# Training the Model
print("\n Training the read out")
pbar = tqdm(enumerate(range(logistic_epochs)))
for epoch, _ in pbar:
    avg_loss = 0
    for i, (s, l) in enumerate(training_pairs):
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(s)
        label = torch.zeros(1, 1, 10).float().to(device_id)
        label[0, 0, l] = 1.0
        loss = criterion(outputs.view(1, 1, -1), label)
        avg_loss += loss.data
        loss.backward()
        optimizer.step()

    pbar.set_description_str(
        "Epoch: %d/%d, Loss: %.4f"
        % (epoch + 1, n_epochs, avg_loss / len(training_pairs))
    )

###########################
# Generating Testing Pairs
###########################

dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, num_workers=n_workers, pin_memory=gpu)
n_iters = n_test
test_pairs = []
pbar = tqdm(enumerate(dataloader_test))
for (i, dataPoint) in pbar:
    if i >= n_iters:
        break
    datum = dataPoint["encoded_image"].view(time, 1, 1, 28, 28).to(device_id)
#    datum = dataPoint["encoded_image"].view(time, 1, 3, 32, 32).to(device_id)
    label = dataPoint["label"]
    pbar.set_description_str("Testing progress: (%d / %d)" % (i, n_iters))

    inputs = {"X": datum}
    network.run(inputs=inputs, time=time, input_time_dim=1)
    test_pairs.append([spikes["Ae"].get("s").sum(0), label])

    
    network.reset_state_variables()

####################
# Test Accuracy
####################

correct, total = 0, 0
for s, label in test_pairs:
    outputs = model(s)
    _, predicted = torch.max(outputs.data.unsqueeze(0), 1)
    total += 1
    correct += int(predicted == label.long().to(device_id))

print(
    "\n Accuracy of the model on %d test images: %.2f %%"
    % (n_iters, 100 * correct / total)
)

