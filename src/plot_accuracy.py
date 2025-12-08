"""
plot_accuracy.py
~~~~~~~~~~~~~~~~

A script to train a neural network and plot accuracy over epochs.
Shows how the network improves as it learns.
"""

import matplotlib.pyplot as plt
import mnist_loader
import network

# Load MNIST data
print("Loading MNIST data...")
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Create network
net = network.Network([784, 30, 10])

# Train network and collect accuracy after each epoch
# We need to modify this to track accuracy per epoch
# For now, let's use a wrapper approach
print("Training network for 30 epochs...")

# Create a custom training loop to track accuracy per epoch
accuracies = []
epochs_list = []

for epoch in range(10):
    # Train for one epoch without test data (faster)
    net.SGD(training_data, epochs=1, mini_batch_size=10, eta=3.0, test_data=None)
    
    # Evaluate on test data
    acc = net.evaluate(test_data)
    accuracies.append(acc)
    epochs_list.append(epoch)
    
    print("Epoch {0}: {1} / 10000 ({2:.1f}%)".format(epoch, acc, acc/100))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(epochs_list, accuracies, 'b-o', linewidth=2, markersize=6)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Correct Predictions (out of 10000)', fontsize=12)
plt.title('Neural Network Accuracy Over Training', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim([7000, 10000])

# Add accuracy percentage on right axis
ax2 = plt.gca().twinx()
ax2.set_ylim([70, 100])
ax2.set_ylabel('Accuracy (%)', fontsize=12)

plt.tight_layout()
plt.savefig('accuracy_plot.png', dpi=150)
print("\nPlot saved as 'accuracy_plot.png'")
plt.show()
