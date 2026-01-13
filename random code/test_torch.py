import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -------------------------------------------------------------
# 1. Device setup
# -------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
if device.type == "cuda":
    print(torch.cuda.get_device_name(0))
    print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1e6:.2f} MB\n")

# -------------------------------------------------------------
# 2. Dataset and Dataloaders
# -------------------------------------------------------------
transform = transforms.ToTensor()

train_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,
)

batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

# -------------------------------------------------------------
# 3. Model definition
# -------------------------------------------------------------
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

model = NeuralNetwork().to(device)
print(model, "\n")

# -------------------------------------------------------------
# 4. Loss and optimizer
# -------------------------------------------------------------
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# -------------------------------------------------------------
# 5. Training and testing loops
# -------------------------------------------------------------
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 200 == 0:
            current = (batch + 1) * len(X)
            if device.type == "cuda":
                gpu_mem = torch.cuda.memory_allocated() / 1e6
                print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}] | GPU mem: {gpu_mem:>7.1f} MB")
            else:
                print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Accuracy: {100*correct:>0.1f}%, Avg loss: {test_loss:>8f}\n")

# -------------------------------------------------------------
# 6. Training loop with timing
# -------------------------------------------------------------
epochs = 5
start_time = time.time()

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model, loss_fn)

total_time = time.time() - start_time
print(f"✅ Done in {total_time:.2f} seconds\n")

# -------------------------------------------------------------
# 7. Save and reload model
# -------------------------------------------------------------
torch.save(model.state_dict(), "model.pth")
print("Saved model to model.pth")

model2 = NeuralNetwork().to(device)
model2.load_state_dict(torch.load("model.pth"))
print("Reloaded model successfully.")


model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
    