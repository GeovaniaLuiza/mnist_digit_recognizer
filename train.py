import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from model import MLP

# carregar CSV
df = pd.read_csv("data/digit-recognizer/train.csv")

X = df.drop("label", axis=1).values / 255.0
y = df["label"].values

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

model = MLP()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):
    optimizer.zero_grad()

    outputs = model(X)
    loss = loss_fn(outputs, y)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")

# salvar modelo
torch.save(model.state_dict(), "models/mnist_pytorch.pth")

print("Modelo salvo com sucesso!")