import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# =====================================================
# CUSTOM PRUNABLE LINEAR LAYER
# =====================================================

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        # Standard trainable weights
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01
        )

        self.bias = nn.Parameter(
            torch.zeros(out_features)
        )

        # Learnable gate parameters
        self.gate_scores = nn.Parameter(
            torch.randn(out_features, in_features)
        )

    def forward(self, x):
        # Convert scores to gate values (0 to 1)
        gates = torch.sigmoid(self.gate_scores)

        # Apply gates to weights
        pruned_weights = self.weight * gates

        return F.linear(x, pruned_weights, self.bias)

    def get_gate_values(self):
        return torch.sigmoid(self.gate_scores)


# =====================================================
# SELF PRUNING NETWORK
# =====================================================

class SelfPruningNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = PrunableLinear(32 * 32 * 3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def sparsity_loss(self):
        """
        L1-style sparsity penalty
        """
        total_loss = 0

        for module in self.modules():
            if isinstance(module, PrunableLinear):
                total_loss += module.get_gate_values().sum()

        return total_loss

    def calculate_sparsity(self, threshold=1e-2):
        """
        Calculates percentage of pruned weights
        """
        total = 0
        pruned = 0

        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates = module.get_gate_values()

                total += gates.numel()
                pruned += (gates < threshold).sum().item()

        return 100 * pruned / total

    def collect_gate_values(self):
        values = []

        for module in self.modules():
            if isinstance(module, PrunableLinear):
                values.extend(
                    module.get_gate_values()
                    .detach()
                    .cpu()
                    .flatten()
                    .tolist()
                )

        return values


# =====================================================
# EVALUATION FUNCTION
# =====================================================

def evaluate_model(model, test_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


# =====================================================
# TRAINING FUNCTION
# =====================================================

def train_model(lambda_value=0.01, epochs=5, batch_size=128, lr=1e-3):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    # CIFAR-10 Dataset
    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = SelfPruningNet().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining with lambda = {lambda_value}\n")

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            classification_loss = criterion(outputs, labels)
            sparsity_penalty = model.sparsity_loss()

            total_loss = (
                classification_loss
                + lambda_value * sparsity_penalty
            )

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        average_loss = running_loss / len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {average_loss:.4f}"
        )

    accuracy = evaluate_model(
        model,
        test_loader,
        device
    )

    sparsity = model.calculate_sparsity()

    print("\nFINAL RESULTS")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Sparsity Level: {sparsity:.2f}%")

    return model, accuracy, sparsity


# =====================================================
# PLOT FUNCTION
# =====================================================

def plot_gate_distribution(model, lambda_value):
    values = model.collect_gate_values()

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=50)

    plt.title(
        f"Gate Distribution (lambda={lambda_value})"
    )
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")

    filename = (
        f"gate_distribution_lambda_{lambda_value}.png"
    )

    plt.savefig(filename)
    plt.show()

    print(f"Saved plot: {filename}")


# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    lambda_values = [0.001, 0.01, 0.1]

    final_results = []

    for lam in lambda_values:
        model, accuracy, sparsity = train_model(
            lambda_value=lam,
            epochs=5
        )

        final_results.append(
            (lam, accuracy, sparsity)
        )

        plot_gate_distribution(
            model,
            lam
        )

    print("\nSUMMARY TABLE")
    print("Lambda | Accuracy | Sparsity %")
    print("-" * 40)

    for row in final_results:
        print(
            f"{row[0]:<7} | "
            f"{row[1]:<8.2f} | "
            f"{row[2]:<10.2f}"
        )