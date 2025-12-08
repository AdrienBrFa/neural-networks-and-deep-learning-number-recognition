"""
test_network2.py
~~~~~~~~~~~~~~~~

A script to test the improved network2.py with MNIST data.
Shows regularization in action and plots training progress.
"""

import matplotlib.pyplot as plt
import mnist_loader
import network2

# Load MNIST data
print("Loading MNIST data...")
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Create network with cross-entropy cost (better than quadratic)
print("Creating network with cross-entropy cost...")
net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)

# Train with regularization (lmbda=5.0)
# lmbda is the regularization parameter (avoids overfitting)
print("\nTraining network2 with regularization (lmbda=5.0)...")
print("This will track both training and evaluation accuracy:\n")

evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net.SGD(
    training_data, 
    epochs=30, 
    mini_batch_size=10,
    eta=0.5,
    lmbda=5.0,  # L2 regularization parameter
    evaluation_data=validation_data,
    monitor_evaluation_accuracy=True,
    monitor_training_accuracy=True,
    monitor_evaluation_cost=True,
    monitor_training_cost=True
)

print("\n" + "="*50)
print("Training Complete!")
print("="*50)
print(f"\nFinal Validation Accuracy: {evaluation_accuracy[-1]} / 10000")
print(f"Final Training Accuracy: {training_accuracy[-1]} / 50000")

# Test on test data
print("\nEvaluating on test data...")
test_accuracy = net.accuracy(test_data)
print(f"Test Accuracy: {test_accuracy} / 10000 ({test_accuracy/100:.1f}%)")

# Save the network
print("\nSaving network to 'network2_trained.json'...")
net.save("network2_trained.json")
print("Done!")

# Plot training and evaluation metrics
print("\nGenerating plots...")
epochs_range = range(len(evaluation_accuracy))

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Accuracy over epochs
ax1.plot(epochs_range, training_accuracy, 'b-o', label='Training', markersize=4)
ax1.plot(epochs_range, evaluation_accuracy, 'r-s', label='Validation', markersize=4)
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Correct Predictions (out of 50000/10000)', fontsize=11)
ax1.set_title('Training vs Validation Accuracy', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Cost over epochs
ax2.plot(epochs_range, training_cost, 'b-o', label='Training', markersize=4)
ax2.plot(epochs_range, evaluation_cost, 'r-s', label='Validation', markersize=4)
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Cost', fontsize=11)
ax2.set_title('Training vs Validation Cost', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Training accuracy only
ax3.plot(epochs_range, training_accuracy, 'b-o', linewidth=2, markersize=5)
ax3.set_xlabel('Epoch', fontsize=11)
ax3.set_ylabel('Correct Predictions', fontsize=11)
ax3.set_title('Training Data Accuracy Over Time', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Validation accuracy only
ax4.plot(epochs_range, evaluation_accuracy, 'r-s', linewidth=2, markersize=5)
ax4.set_xlabel('Epoch', fontsize=11)
ax4.set_ylabel('Correct Predictions', fontsize=11)
ax4.set_title('Validation Data Accuracy Over Time', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('network2_training_analysis.png', dpi=150)
print("Plot saved as 'network2_training_analysis.png'")
plt.show()
