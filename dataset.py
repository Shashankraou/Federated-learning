import tensorflow as tf
import numpy as np

def load_client_data(client_id, num_clients=5):
    """
    Improved data distribution: Each client gets all classes but with imbalance
    This is more realistic than giving only 2 classes per client
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = y_train.flatten(), y_test.flatten()

    # Use Dirichlet distribution for realistic non-IID data
    n_classes = 10
    alpha = 0.5  # Controls non-IID level (0.5 = moderately non-IID)
    
    client_data_indices = [[] for _ in range(num_clients)]
    
    for k in range(n_classes):
        # Get all indices for this class
        idx_k = np.where(y_train == k)[0]
        np.random.shuffle(idx_k)
        
        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Split indices according to proportions
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        client_class_indices = np.split(idx_k, proportions)
        
        # Assign to each client
        for client_idx, indices in enumerate(client_class_indices):
            client_data_indices[client_idx].extend(indices)
    
    # Get this client's data
    client_indices = client_data_indices[client_id]
    np.random.shuffle(client_indices)
    
    x_train_client = x_train[client_indices]
    y_train_client = y_train[client_indices]
    
    # Print statistics
    print(f"Client {client_id}: {len(x_train_client)} samples")
    class_counts = np.bincount(y_train_client, minlength=10)
    print(f"  Class distribution: {class_counts}")
    
    return (x_train_client, y_train_client), (x_test, y_test)