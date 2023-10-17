import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define a simple autoregressive model for prediction
class AutoRegressiveModel(nn.Module):
    def __init__(self, input_dim):
        super(AutoRegressiveModel, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)  # Output dimension is the same as input for multivariate data

    def forward(self, x):
        x = x.to(torch.float32)
        return self.linear(x)

# Define a simple feedforward neural network for the discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.to(torch.float32)
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return torch.sigmoid(x)

# Function to compute predictive score
def compute_predictive_score(real_data, synthetic_data, num_epochs=100):
    synthetic_data = synthetic_data.float()
    # Assume real_data and synthetic_data are tensors of shape (batch_size, sequence_length, num_variables)
    input_dim = real_data.size(2)  # Number of variables
    sequence_length = real_data.size(1)  # Sequence length

    # Flatten the data for autoregressive modeling
    real_data_flat = real_data.view(-1, input_dim)
    synthetic_data_flat = synthetic_data.view(-1, input_dim)

    # Instantiate and train the autoregressive model
    ar_model = AutoRegressiveModel(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(ar_model.parameters(), lr=0.001)

    # Train on real data
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        real_predictions = ar_model(real_data_flat)
        real_predictions = real_predictions.view(-1, sequence_length, input_dim)[:, :-1]  # Predict next step
        loss = criterion(real_predictions, real_data[:, 1:].float())  # Align dimensions for loss calculation
        loss.backward()
        optimizer.step()

    # Evaluate on synthetic data
    with torch.no_grad():
        synthetic_predictions = ar_model(synthetic_data_flat)
        synthetic_predictions = synthetic_predictions.view(-1, sequence_length, input_dim)[:, :-1]  # Align dimensions

    # Compute mean squared error as the predictive score
    predictive_score = criterion(synthetic_predictions, synthetic_data[:, 1:])
    return predictive_score.item()

# Function to compute discriminative score
def compute_discriminative_score(real_data, synthetic_data, num_epochs=100):
    # Flatten and concatenate real and synthetic data
    combined_data_flat = torch.cat((real_data.view(-1, real_data.size(2)), 
                                    synthetic_data.view(-1, synthetic_data.size(2))), dim=0)

    # Labels: 1 for real data, 0 for synthetic data
    real_labels = torch.ones(real_data.size(0) * real_data.size(1), 1)
    synthetic_labels = torch.zeros(synthetic_data.size(0) * synthetic_data.size(1), 1)
    labels = torch.cat((real_labels, synthetic_labels), dim=0)

    # Shuffle the data and labels
    indices = torch.randperm(combined_data_flat.size(0))
    combined_data_flat = combined_data_flat[indices]
    labels = labels[indices]

    # Instantiate and train the discriminator
    discriminator = Discriminator(real_data.size(2))
    criterion = nn.BCELoss()
    optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predictions = discriminator(combined_data_flat)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

    # Evaluate on real data (discrimination accuracy)
    with torch.no_grad():
        real_predictions = discriminator(real_data.view(-1, real_data.size(2)))
        real_accuracy = ((real_predictions > 0.5).float() == real_labels).float().mean().item()

    return real_accuracy