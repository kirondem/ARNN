import torch
import torch.nn as nn
import torch.optim as optim

# 1. Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, encoding_dim),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_size),
            nn.ReLU(True)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model, loss, and optimizer
input_size = 512
encoding_dim = 256
model = Autoencoder(input_size, encoding_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Generate some random data for training
num_samples = 5000
data = torch.rand(num_samples, input_size).to(device)

# 4. Train the autoencoder
num_epochs = 10
batch_size = 256

for epoch in range(num_epochs):
    for i in range(0, num_samples, batch_size):
        batch_data = data[i:i+batch_size]
        outputs = model(batch_data)
        loss = criterion(outputs, batch_data)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save model weights
torch.save(model.state_dict(), "autoencoder.pth")

# 5. Encode a sample vector to demonstrate the reduced dimensionality
sample_vector = torch.rand(1, input_size).to(device)
encoded_vector = model.encoder(sample_vector)

encoded_vector.shape

print(encoded_vector)
