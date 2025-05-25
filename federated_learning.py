import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import matplotlib.pyplot as plt
import logging
from collections import OrderedDict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load and preprocess EMNIST data
def create_non_iid_data(num_clients=10):
    """Create non-IID data for clients by skewing label distributions."""
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
    client_data = []
    client_ids = [f"client_{i}" for i in range(num_clients)]

    # Load raw EMNIST data
    dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[0])
    images, labels = [], []
    for sample in dataset:
        images.append(sample['pixels'].numpy())
        labels.append(sample['label'].numpy())

    images = np.array(images)
    labels = np.array(labels)

    # Create non-IID partitions
    for i in range(num_clients):
        # Skew labels: each client gets a subset of labels
        primary_labels = [(i % 10), (i % 10 + 1) % 10]  # Client i prefers two labels
        mask = np.isin(labels, primary_labels)
        client_indices = np.where(mask)[0]
        np.random.shuffle(client_indices)

        # Vary data volume (100 to 1000 samples)
        num_samples = np.random.randint(100, 1000)
        client_indices = client_indices[:num_samples]

        # Create client dataset
        client_images = images[client_indices]
        client_labels = labels[client_indices]
        client_dataset = tf.data.Dataset.from_tensor_slices(
            (client_images.reshape(-1, 28, 28, 1), client_labels)
        ).batch(32)
        client_data.append(client_dataset)

    # Create test dataset
    test_dataset = emnist_test.create_tf_dataset_from_all_clients().batch(32)
    return client_data, client_ids, test_dataset

# Define Keras model
def create_keras_model():
    """Create and compile a CNN model for EMNIST."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    return model

# TFF model function
def model_fn():
    """Convert Keras model to TFF model."""
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model=keras_model,
        input_spec=federated_train_data[0].element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# Evaluate global model on test data
def evaluate_model(server_state, test_dataset):
    """Evaluate the global model on test data."""
    keras_model = create_keras_model()
    tff.learning.state_with_new_model_weights(
        keras_model,
        server_state.model_weights,
        model_fn().trainable_variables
    )
    keras_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    loss, accuracy = keras_model.evaluate(test_dataset, verbose=0)
    return loss, accuracy

# Visualize convergence
def plot_convergence(rounds, accuracies):
    """Plot model convergence over rounds."""
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, accuracies, marker='o')
    plt.title("Global Model Accuracy Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig("convergence_task1.png")
    plt.close()
    logger.info("Convergence plot saved as convergence_task1.png")

# Main function
def main():
    global federated_train_data
    # Initialize data
    federated_train_data, client_ids, test_dataset = create_non_iid_data(num_clients=10)
    logger.info(f"Created non-IID data for {len(client_ids)} clients")

    # Initialize federated process
    iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=model_fn,
        client_optimizer_fn=tf.keras.optimizers.Adam(learning_rate=0.001),
        server_optimizer_fn=tf.keras.optimizers.SGD(learning_rate=1.0)
    )
    server_state = iterative_process.initialize()
    logger.info("Federated process initialized")

    # Training loop
    num_rounds = 10
    accuracies = []
    for round_num in range(1, num_rounds + 1):
        # Select all clients for simplicity (partial participation in next task)
        selected_clients = client_ids
        logger.info(f"Round {round_num}: Training with {len(selected_clients)} clients")
        
        # Run one round
        state, metrics = iterative_process.next(server_state, [federated_train_data[client_ids.index(cid)] for cid in selected_clients])
        server_state = state
        
        # Evaluate on test data
        loss, accuracy = evaluate_model(server_state, test_dataset)
        accuracies.append(accuracy)
        logger.info(f"Round {round_num}: Test Loss = {loss:.4f}, Test Accuracy = {accuracy:.4f}")

    # Plot convergence
    plot_convergence(range(1, num_rounds + 1), accuracies)

if __name__ == "__main__":
    main()