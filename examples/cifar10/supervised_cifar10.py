import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt

import torchvision
from torchvision import transforms
from tqdm import tqdm

from bindsnet.encoding import PoissonEncoder
from bindsnet.datasets import CIFAR10
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.analysis.plotting import (
    plot_input,
    plot_assignments,
    plot_performance,
    plot_weights,
    plot_spikes,
    plot_voltages,
)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_train", type=int, default=600)
parser.add_argument("--n_test", type=int, default=600)
parser.add_argument("--n_clamp", type=int, default=1)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=22.5)
parser.add_argument("--time", type=int, default=500)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=25)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--device_id", type=int, default=0)
parser.set_defaults(plot=False, gpu=False, train=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_train = args.n_train
n_test = args.n_test
n_clamp = args.n_clamp
exc = args.exc
inh = args.inh
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
#TODO: gpu problem
gpu = args.gpu
device_id = args.device_id


num_classes = 10
cifar_shape = (32, 32, 3)

np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

# Sets up Gpu use
if gpu and torch.cuda.is_available():
    torch.cuda.set_device(device_id)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.manual_seed(seed)

if not train:
    update_interval = n_test

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity
per_class = int(n_neurons / num_classes)

# Build Diehl & Cook 2015 network.
network = DiehlAndCook2015(
    n_inpt=32*32*3,
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    dt=dt,
    norm=78.4,
    nu=[0, 1e-2],
    inpt_shape=cifar_shape, 
)

# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(network.layers["Ae"], ["v"], time=time)
inh_voltage_monitor = Monitor(network.layers["Ai"], ["v"], time=time)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")

# Load CIFAR10 data.
train_dataset = CIFAR10(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "CIFAR10"),
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), 
         #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         transforms.Lambda(lambda x: x * intensity)] 
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

# Create a dataloader to iterate and batch data
dataloader_train = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=gpu
)

dataloader_test = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=gpu
)

# Record spikes during the simulation.
spike_record = torch.zeros(update_interval, time, n_neurons)

# Neuron assignments and spike proportions.
assignments = -torch.ones_like(torch.Tensor(n_neurons))
proportions = torch.zeros_like(torch.Tensor(n_neurons, num_classes))
rates = torch.zeros_like(torch.Tensor(n_neurons, num_classes))

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

# Labels to determine neuron assignments and spike proportions and estimate accuracy
labels = torch.empty(update_interval)

spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

if gpu:
    network.to("cuda")

# Train the network.
print("Begin training.\n")

inpt_axes = None
inpt_ims = None
spike_axes = None
spike_ims = None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes = None
voltage_ims = None

pbar = tqdm(enumerate(dataloader_train))
for (i, datum) in pbar:
    if gpu:
        datum = datum.to("cuda")

    if i > n_train:
        break

    image = datum["encoded_image"]
    label = datum["label"]
    pbar.set_description_str("Train progress: (%d / %d)" % (i, n_train))

    #Print training accuracy
    if i % update_interval == 0 and i > 0:
        # Get network predictions.
        all_activity_pred = all_activity(spike_record, assignments, num_classes) 
        proportion_pred = proportion_weighting(
            spike_record, assignments, proportions, num_classes 
        )

        # Compute network accuracy according to available classification strategies.
        accuracy["all"].append(
            100 * torch.sum(labels.long() == all_activity_pred).item() / update_interval
        )
        accuracy["proportion"].append(
            100 * torch.sum(labels.long() == proportion_pred).item() / update_interval
        )

        print(
            "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
            % (accuracy["all"][-1], np.mean(accuracy["all"]), np.max(accuracy["all"]))
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
        assignments, proportions, rates = assign_labels(spike_record, labels, num_classes, rates)

    #Add the current label to the list of labels for this update_interval
    labels[i % update_interval] = label[0]

    # Run the network on the input.
    choice = np.random.choice(int(n_neurons / num_classes), size=n_clamp, replace=False) 
    clamp = {"Ae": per_class * label.long() + torch.Tensor(choice).long()}
    inputs = {"X": image.view(time, 1, 32, 32, 3)}
    network.run(inputs=inputs, time=time, clamp=clamp)

    # Get voltage recording.
    exc_voltages = exc_voltage_monitor.get("v")
    inh_voltages = inh_voltage_monitor.get("v")

    # Add to spikes recording.
    spike_record[i % update_interval] = spikes["Ae"].get("s").view(time, n_neurons)

    # Optionally plot various simulation information.
    if plot:
        inpt = inputs["X"].view(time, 32*32*3).sum(0).view(32, 32* 3)
        input_exc_weights = network.connections[("X", "Ae")].w
        square_weights = get_square_weights(
            input_exc_weights.view(32*32*3, n_neurons), n_sqrt, (32, 32* 3)
        )
        square_assignments = get_square_assignments(assignments, n_sqrt)
        voltages = {"Ae": exc_voltages, "Ai": inh_voltages}

        inpt_axes, inpt_ims = plot_input(
            image.sum(1).view(32, 32, 3), inpt, label=label, axes=inpt_axes, ims=inpt_ims 
        )
        spike_ims, spike_axes = plot_spikes(
            {layer: spikes[layer].get("s").view(time, 1, -1) for layer in spikes},
            ims=spike_ims,
            axes=spike_axes,
        )
        weights_im = plot_weights(square_weights, im=weights_im)
        assigns_im = plot_assignments(square_assignments, im=assigns_im)
        perf_ax = plot_performance(accuracy, ax=perf_ax)
        voltage_ims, voltage_axes = plot_voltages(
            voltages, ims=voltage_ims, axes=voltage_axes
        )
        plt.pause(1e-8)

    network.reset_state_variables()  # Reset state variables.

print("Progress: %d / %d \n" % (n_train, n_train))
print("Training complete.\n")

#TODO: Add testing loop

# Record spikes during the simulation.
spike_record_test = torch.zeros(update_interval, time, n_neurons)

# Neuron assignments and spike proportions.
assignments_test = -torch.ones_like(torch.Tensor(n_neurons))
proportions_test = torch.zeros_like(torch.Tensor(n_neurons, num_classes))
rates_test = torch.zeros_like(torch.Tensor(n_neurons, num_classes))

test_accuracy = {"all": [], "proportion": []}


pbar = tqdm(enumerate(dataloader_test))
for (i, datum) in pbar:
    if gpu:
        datum = datum.to("cuda")

    if i > n_test:
        break

    image = datum["encoded_image"]
    label = datum["label"]
    pbar.set_description_str("Testing progress: (%d / %d)" % (i, n_test))

    #Print training accuracy
    if i % update_interval == 0 and i > 0:
        # Get network predictions.
        all_activity_pred = all_activity(spike_record_test, assignments_test, num_classes) 
        proportion_pred = proportion_weighting(
            spike_record_test, assignments_test, proportions_test, num_classes 
        )

        # Compute network accuracy according to available classification strategies.
        test_accuracy["all"].append(
            100 * torch.sum(labels.long() == all_activity_pred).item() / update_interval
        )
        test_accuracy["proportion"].append(
            100 * torch.sum(labels.long() == proportion_pred).item() / update_interval
        )

        print(
            "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
            % (test_accuracy["all"][-1], np.mean(test_accuracy["all"]), np.max(test_accuracy["all"]))
        )
        print(
            "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f (best)\n"
            % (
                test_accuracy["proportion"][-1],
                np.mean(test_accuracy["proportion"]),
                np.max(test_accuracy["proportion"]),
            )
        )

        # Assign labels to excitatory layer neurons.
        assignments_test, proportions_test, rates_test = assign_labels(spike_record_test, labels, num_classes, rates_test)

    #Add the current label to the list of labels for this update_interval
    labels[i % update_interval] = label[0]

    # Run the network on the input.
    inputs = {"X": image.view(time, 1, 32, 32, 3)}
    network.run(inputs=inputs, time=time)

    # Get voltage recording.
    exc_voltages = exc_voltage_monitor.get("v")
    inh_voltages = inh_voltage_monitor.get("v")

    # Add to spikes recording.
    spike_record_test[i % update_interval] = spikes["Ae"].get("s").view(time, n_neurons)

    # Optionally plot various simulation information.
    if plot:
        inpt = inputs["X"].view(time, 32*32*3).sum(0).view(32, 32* 3)
        input_exc_weights = network.connections[("X", "Ae")].w
        square_weights = get_square_weights(
            input_exc_weights.view(32*32*3, n_neurons), n_sqrt, (32, 32* 3)
        )
        square_assignments = get_square_assignments(assignments_test, n_sqrt)
        voltages = {"Ae": exc_voltages, "Ai": inh_voltages}

        inpt_axes, inpt_ims = plot_input(
            image.sum(1).view(32, 32, 3), inpt, label=label, axes=inpt_axes, ims=inpt_ims 
        )
        spike_ims, spike_axes = plot_spikes(
            {layer: spikes[layer].get("s").view(time, 1, -1) for layer in spikes},
            ims=spike_ims,
            axes=spike_axes,
        )
        weights_im = plot_weights(square_weights, im=weights_im)
        assigns_im = plot_assignments(square_assignments, im=assigns_im)
        perf_ax = plot_performance(test_accuracy, ax=perf_ax)
        voltage_ims, voltage_axes = plot_voltages(
            voltages, ims=voltage_ims, axes=voltage_axes
        )
        plt.pause(1e-8)

    network.reset_state_variables()  # Reset state variables.

print("Progress: %d / %d \n" % (n_train, n_train))
print("Training complete.\n")
