import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# Cihaz ayarı
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Kullanılan cihaz: {device}")

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder katmanları
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # mu katmanı
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # logvar katmanı
        
        # Decoder katmanları
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        """Encoder: x -> mu, logvar"""
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = mu + std * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decoder: z -> x_reconstructed"""
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    """VAE loss fonksiyonu: BCE + KLD"""
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD

def train_vae(model, train_loader, optimizer, epoch):
    """Bir epoch için eğitim"""
    model.train()
    train_loss = 0
    
    for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    return train_loss / len(train_loader.dataset)

def test_vae(model, test_loader):
    """Test seti üzerinde değerlendirme"""
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
    
    return test_loss / len(test_loader.dataset)

def generate_samples(model, num_samples=64):
    """Latent space'den yeni örnekler üretme"""
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, 20).to(device)  # latent_dim = 20
        samples = model.decode(z).cpu()
        return samples

def plot_reconstruction(model, test_loader, n=10):
    """Orijinal ve yeniden yapılandırılmış görüntüleri gösterme"""
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data.to(device)
        recon_batch, _, _ = model(data)
        
    fig, axes = plt.subplots(2, n, figsize=(20, 4))
    for i in range(n):
        # Orijinal görüntü
        axes[0][i].imshow(data[i].cpu().squeeze(), cmap='gray')
        axes[0][i].set_title('Orijinal')
        axes[0][i].axis('off')
        
        # Yeniden yapılandırılmış görüntü
        axes[1][i].imshow(recon_batch[i].cpu().view(28, 28), cmap='gray')
        axes[1][i].set_title('Yeniden Yapılandırılmış')
        axes[1][i].axis('off')
    
    plt.tight_layout()
    plt.savefig('reconstruction_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_generated_samples(model, n=8):
    """Üretilen yeni örnekleri gösterme"""
    samples = generate_samples(model, n*n)
    
    fig, axes = plt.subplots(n, n, figsize=(10, 10))
    for i in range(n):
        for j in range(n):
            axes[i][j].imshow(samples[i*n + j].view(28, 28), cmap='gray')
            axes[i][j].axis('off')
    
    plt.suptitle('VAE ile Üretilen Yeni Görüntüler', fontsize=16)
    plt.tight_layout()
    plt.savefig('generated_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_latent_space(model, test_loader, num_classes=10):
    """2D latent space görselleştirmesi (latent_dim=2 olması gerekir)"""
    if model.fc21.out_features != 2:
        print("Latent space görselleştirmesi için latent_dim=2 olmalıdır")
        return
    
    model.eval()
    z_mean = []
    labels = []
    
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            mu, _ = model.encode(data.view(-1, 784))
            z_mean.append(mu.cpu().numpy())
            labels.append(label.numpy())
    
    z_mean = np.concatenate(z_mean)
    labels = np.concatenate(labels)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title('VAE Latent Space Representation')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.savefig('latent_space.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Hyperparameters
    batch_size = 128
    learning_rate = 1e-3
    epochs = 20
    latent_dim = 20
    
    # Veri yükleme ve preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST('./MNIST', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST('./MNIST', train=False, download=False, transform=transform)

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model oluşturma
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Eğitim
    train_losses = []
    test_losses = []
    
    print("VAE Eğitimi Başlıyor...")
    for epoch in range(1, epochs + 1):
        train_loss = train_vae(model, train_loader, optimizer, epoch)
        test_loss = test_vae(model, test_loader)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        print(f'Epoch {epoch:2d}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        
        # Her 5 epoch'ta bir görüntü kaydetme
        if epoch % 5 == 0:
            plot_generated_samples(model)
    
    # Sonuçları kaydetme
    torch.save(model.state_dict(), 'vae_model.pth')
    
    # Loss grafiği
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VAE Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Yeniden yapılandırma karşılaştırması
    plot_reconstruction(model, test_loader)
    
    # Son üretilen örnekler
    plot_generated_samples(model)
    
    print("Eğitim tamamlandı!")

if __name__ == "__main__":
    main()
