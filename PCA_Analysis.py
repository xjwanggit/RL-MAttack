import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)  # For reproducibility
user_embedding = np.random.rand(1, 64)

# Generate random item embeddings before attack (5 items, 64-dimensional)
item_embedding_before = np.random.rand(5, 64)

# Combine embeddings for consistent PCA fitting
combined_embeddings = np.vstack([user_embedding, item_embedding_before])

# Apply PCA to reduce to 2D
pca = PCA(n_components=2)
combined_embeddings_2d = pca.fit_transform(combined_embeddings)

# Split the reduced data back into user and item embeddings
user_embedding_2d = combined_embeddings_2d[:1]  # User embedding in 2D
item_embedding_before_2d = combined_embeddings_2d[1:]  # Item embeddings before attack in 2D

plt.figure(figsize=(10, 8))
origin = (0, 0)  # Start all vectors from (0, 0)

# Plot the user embedding vector
plt.arrow(origin[0], origin[1], user_embedding_2d[0, 0], user_embedding_2d[0, 1],
          head_width=0.08, head_length=0.1, fc='blue', ec='blue', alpha=0.8, linewidth=1.5, label="User Embedding")

# Define step labels
step_labels = ["STEP 1", "STEP 2", "STEP 3", "STEP 4", "STEP 5"]

# Plot item embeddings before attack in green and add step labels
for i, pos_before in enumerate(item_embedding_before_2d):
    # Plot the arrow for item embedding
    plt.arrow(origin[0], origin[1], pos_before[0], pos_before[1],
              head_width=0.05, head_length=0.07, fc='green', ec='green', alpha=0.6, linewidth=1.2)

    # Compute offset for annotation
    offset_x = pos_before[0] * 0.15  # Adjust distance factor as needed
    offset_y = pos_before[1] * 0.15

    # Annotate with step labels, adjusted position
    plt.text(pos_before[0] + offset_x, pos_before[1] + offset_y, step_labels[i],
             color="green", fontsize=10, ha='center', va='center')

# Add axis lines and a grid for better readability
plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)

# Labels and title
plt.xlabel('PCA Component 1', fontsize=12)
plt.ylabel('PCA Component 2', fontsize=12)
plt.title('2D PCA Representation of User and Item Embeddings', fontsize=14)
plt.legend(['User Embedding', 'Item Embeddings (Before Attack)'], loc='upper left', fontsize=10)
plt.grid(color='gray', linestyle=':', linewidth=0.5)

# Save the plot
plt.savefig('1.jpg', dpi=300, bbox_inches='tight')