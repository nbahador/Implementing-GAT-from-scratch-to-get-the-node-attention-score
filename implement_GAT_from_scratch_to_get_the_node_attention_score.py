
# implement GAT from scratch to get the node attention score

import torch
import torch.nn as nn

class GAT(nn.Module):
    def __init__(self, num_nodes, num_features, num_classes, num_attention_heads, dropout=0.8):
        super(GAT, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_heads = nn.ModuleList([nn.Linear(num_features, num_features, bias=False) for _ in range(num_attention_heads)])
        self.out_att = nn.Linear(num_attention_heads * num_features, num_classes, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def l2_regularization(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += param.pow(2).sum()
        return l2_loss

    def forward(self, x, adj):
        # x: node features (batch_size, num_nodes, num_features)
        # adj: adjacency matrix (batch_size, num_nodes, num_nodes)
        h = x
        h_out = h.clone()
        aaa = enumerate(self.attention_heads)

        for i, attention_head in enumerate(self.attention_heads[:-1]):
            # Apply attention head to the node features
            h_i = attention_head(h)
            # Calculate attention weights
            attention_weights = torch.matmul(h_i, h.transpose(1, 2))
            attention_weights = attention_weights / torch.sqrt(torch.tensor(h.size(1)).float())
            attention_weights = torch.exp(attention_weights) * adj
            attention_weights = attention_weights / attention_weights.sum(dim=2, keepdim=True)
            self.attention_weights = attention_weights
            # Apply attention weights to the node features
            h = torch.matmul(attention_weights, h)
            # Concatenate the output of the attention head with the output from previous heads
            zzz = torch.cat((h_out, h), dim=-1)
            h_out2 = torch.squeeze(zzz, dim=-1)
            h_out = h_out2


        # Apply dropout and final linear layer
        h_out3 = self.dropout(h_out)
        logits = self.out_att(h_out3)
        return logits


# Load node features and adjacency matrix
import numpy as np
# Set number of nodes and features
num_nodes = 8
num_features = 3
num_classes = 8
num_epochs = 100


batch_size = 1

# Generate random node features
#x = np.random.rand(num_nodes, num_features)

beta = 1   # perturbation factor

x = np.array(([0.5*beta,-0.1*beta,0.3*beta],
                       [0.2,0.1,0.7],
                       [-0.5,0.7,-0.1],
                       [-0.1,-0.6,0.4],
                       [0.3,-0.5,-0.2],
                       [0.1,-0.1,-0.4],
                       [0.3,0.8,-0.1],
                       [0.1,-0.2,0.2]), dtype=float)

b10 = 0.1295
a11 = 517.0544
b11 = 115.5967
a12 = 4.2614
b12 = 4.6361
a13 = 1.3083
b13 = 1.4428
a14 = 2.7480
b14 = 3.1052
a21 = 0.9492
b20 = -0.5262
a22 = 2.6331
b21 = 3.7399
a41 = 239.0092
b22 = 9.6729
a42 = 1.7819
b40 = 0.1595
a43 = 3.6549
b41 = 26.8926
a44 = 5.2346
b42 = 1.9330
a31 = 191.8224
b43 = 4.0660
a32 = 49.4899
b44 = 5.9926
b31 = 0.9688
b30 = 3.3
b32 = 0.2043

# Generate random adjacency matrix
adj = np.array([[0,1,0,0,0,1,0,0],
              [0,0,1,0,0,0,1,0],
              [0,0,0,1,0,0,0,1],
              [-a14,a13,-a12,-a11,0,0,-a22,-a21],
              [0,1,0,0,0,1,0,0],
              [0,0,1,0,0,0,1,0],
              [0,0,0,1,0,0,0,1],
              [0,0,-a32,-a31,-a44,-a43,-a42,-a41]])

#adj = np.random.randint(0, 2, size=(num_nodes, num_nodes))

np.fill_diagonal(adj, 0)  # Set diagonal to 0

# Add batch size dimension to x and adj
x = np.expand_dims(x, 0)  # shape: (batch_size, num_nodes, num_features)
adj = np.expand_dims(adj, 0)  # shape: (batch_size, num_nodes, num_nodes)

# Repeat x and adj to match batch size
x = np.repeat(x, batch_size, axis=0)  # shape: (batch_size, num_nodes, num_features)
adj = np.repeat(adj, batch_size, axis=0)  # shape: (batch_size, num_nodes, num_nodes)

# Convert to PyTorch tensors
x = torch.from_numpy(x).float()
adj = torch.from_numpy(adj).float()

# Set maximum number of epochs and number of epochs without improvement to stop
max_epochs = 100
patience = 5

# Initialize model and optimizer
model = GAT(num_nodes=x.size(1), num_features=x.size(2), num_classes=num_classes, num_attention_heads=2)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# Initialize early stopping counter
counter = 0

# Initialize best loss to a large value
best_loss = float("inf")

# Loop over epochs
for epoch in range(max_epochs):
    # Generate random labels
    labels = np.random.randint(0, num_classes, size=num_nodes)

    # Convert to PyTorch tensor
    labels = torch.from_numpy(labels).long()

    # Ensure labels have the correct batch size
    labels = labels.unsqueeze(0).repeat(x.size(0), 1)

    # Forward pass
    logits = model(x, adj)
    #loss = nn.functional.cross_entropy(logits, labels)
    loss = nn.functional.cross_entropy(logits, labels) + 0.01 * model.l2_regularization()

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check if loss is the best we've seen so far
    if loss < best_loss:
        # Update best loss and reset counter
        best_loss = loss
        counter = 0
    else:
        # Increment counter
        counter += 1

    # Check if we have reached the patience limit
    if counter >= patience:
        print("Early stopping at epoch", epoch)
        break

    # Print loss
    print("Epoch: {}, Loss: {:.4f}".format(epoch+1, loss.item()))

output = model.attention_weights
output = torch.squeeze(output, dim=0)
array = output.H
array = array.detach().cpu().numpy()



adj2 = torch.squeeze(adj, dim=0).numpy()
degree_matrix = np.diag(np.sum(adj2, axis = 0))
degree_matrix_inv = np.linalg.inv(degree_matrix)
result = np.multiply(array, adj2)
result2 = np.sum(result, axis = 0).reshape([-1,1])
dddd = np.diag(degree_matrix_inv).reshape([-1,1])
node_scores = np.multiply(result2,dddd)

#objects = ('Node 1', 'Node 2', 'Node 3', 'Node 4', 'Node 5', 'Node 6', 'Node 7', 'Node 8')
#y_pos = np.arange(len(objects)).reshape([-1,1])
#plt.bar(y_pos, node_scores, align='center', alpha=0.5)
#plt.xticks(y_pos, objects)
#plt.ylabel('Attention Coefficients')
#plt.title('After Perturbation at Node 1')
#plt.show()

