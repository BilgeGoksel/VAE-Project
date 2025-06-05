import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class StandardAutoencoder(nn.Module):
    """Karşılaştırma için standart autoencoder"""
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(StandardAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.view(-1, 784)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_standard_ae(model, train_loader, optimizer, epoch):
    """Standart autoencoder eğitimi"""
    model.train()
    train_loss = 0
    
    for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f'Standard AE Epoch {epoch}')):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch = model(data)
        loss = F.binary_cross_entropy(recon_batch, data.view(-1, 784), reduction='sum')
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    return train_loss / len(train_loader.dataset)

def test_standard_ae(model, test_loader):
    """Standart autoencoder test"""
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon_batch = model(data)
            test_loss += F.binary_cross_entropy(recon_batch, data.view(-1, 784), reduction='sum').item()
    
    return test_loss / len(test_loader.dataset)

def compare_models():
    """VAE ve Standart Autoencoder karşılaştırması"""
    
    # Hyperparameters
    batch_size = 128
    learning_rate = 1e-3
    epochs = 20
    latent_dim = 20
    
    # Veri yükleme
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./MNIST', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST('./MNIST', train=False, download=False, transform=transform)

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Modeller
    from vae_implementation import VAE, device
    
    vae_model = VAE(latent_dim=latent_dim).to(device)
    std_ae_model = StandardAutoencoder(latent_dim=latent_dim).to(device)
    
    vae_optimizer = optim.Adam(vae_model.parameters(), lr=learning_rate)
    std_ae_optimizer = optim.Adam(std_ae_model.parameters(), lr=learning_rate)
    
    # Eğitim sonuçları
    vae_train_losses = []
    vae_test_losses = []
    std_ae_train_losses = []
    std_ae_test_losses = []
    
    print("Model Karşılaştırması Başlıyor...")
    
    for epoch in range(1, epochs + 1):
        # VAE eğitimi
        from vae_implementation import train_vae, test_vae
        vae_train_loss = train_vae(vae_model, train_loader, vae_optimizer, epoch)
        vae_test_loss = test_vae(vae_model, test_loader)
        
        # Standart AE eğitimi
        std_ae_train_loss = train_standard_ae(std_ae_model, train_loader, std_ae_optimizer, epoch)
        std_ae_test_loss = test_standard_ae(std_ae_model, test_loader)
        
        vae_train_losses.append(vae_train_loss)
        vae_test_losses.append(vae_test_loss)
        std_ae_train_losses.append(std_ae_train_loss)
        std_ae_test_losses.append(std_ae_test_loss)
        
        print(f'Epoch {epoch:2d}:')
        print(f'  VAE - Train: {vae_train_loss:.4f}, Test: {vae_test_loss:.4f}')
        print(f'  Std AE - Train: {std_ae_train_loss:.4f}, Test: {std_ae_test_loss:.4f}')
    
    # Karşılaştırma grafikleri
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss karşılaştırması
    ax1.plot(vae_train_losses, label='VAE Train', color='blue')
    ax1.plot(std_ae_train_losses, label='Standard AE Train', color='red')
    ax1.set_title('Training Loss Comparison')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(vae_test_losses, label='VAE Test', color='blue')
    ax2.plot(std_ae_test_losses, label='Standard AE Test', color='red')
    ax2.set_title('Test Loss Comparison')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Yeniden yapılandırma kalitesi karşılaştırması
    with torch.no_grad():
        test_data, _ = next(iter(test_loader))
        test_data = test_data.to(device)
        
        # VAE reconstruction
        vae_recon, _, _ = vae_model(test_data)
        
        # Standard AE reconstruction
        std_ae_recon = std_ae_model(test_data)
        
        # İlk 5 görüntüyü göster
        for i in range(5):
            # Orijinal
            ax3.imshow(test_data[i].cpu().squeeze(), cmap='gray')
            ax3.set_title('Original Images')
            ax3.axis('off')
            
            # VAE vs Standard AE reconstruction karşılaştırması
            if i == 0:  # Sadece ilk görüntü için subplot
                fig2, axes = plt.subplots(3, 5, figsize=(15, 9))
                for j in range(5):
                    axes[0][j].imshow(test_data[j].cpu().squeeze(), cmap='gray')
                    axes[0][j].set_title(f'Original {j+1}')
                    axes[0][j].axis('off')
                    
                    axes[1][j].imshow(vae_recon[j].cpu().view(28, 28), cmap='gray')
                    axes[1][j].set_title(f'VAE Recon {j+1}')
                    axes[1][j].axis('off')
                    
                    axes[2][j].imshow(std_ae_recon[j].cpu().view(28, 28), cmap='gray')
                    axes[2][j].set_title(f'Std AE Recon {j+1}')
                    axes[2][j].axis('off')
                
                plt.tight_layout()
                plt.savefig('model_comparison_reconstruction.png', dpi=300, bbox_inches='tight')
                plt.show()
    
    plt.tight_layout()
    plt.savefig('model_comparison_losses.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Nicel karşılaştırma
    print("\n=== MODEL KARŞILAŞTIRMA SONUÇLARI ===")
    print(f"Final VAE Test Loss: {vae_test_losses[-1]:.4f}")
    print(f"Final Standard AE Test Loss: {std_ae_test_losses[-1]:.4f}")
    
    # MSE hesaplama
    with torch.no_grad():
        vae_mse = F.mse_loss(vae_recon, test_data.view(-1, 784))
        std_ae_mse = F.mse_loss(std_ae_recon, test_data.view(-1, 784))
        
        print(f"VAE Reconstruction MSE: {vae_mse:.6f}")
        print(f"Standard AE Reconstruction MSE: {std_ae_mse:.6f}")

if __name__ == "__main__":
    compare_models()
