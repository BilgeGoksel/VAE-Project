import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def evaluate_model_quality(model, test_loader, device, model_name="VAE"):
    """Model kalitesini değerlendiren kapsamlı fonksiyon"""
    model.eval()
    
    total_loss = 0
    total_mse = 0
    total_samples = 0
    
    reconstruction_losses = []
    mse_losses = []
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            batch_size = data.size(0)
            
            if model_name == "VAE":
                recon_batch, mu, logvar = model(data)
                # VAE loss hesaplama
                BCE = F.binary_cross_entropy(recon_batch, data.view(-1, 784), reduction='mean')
                KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = BCE + KLD
                reconstruction_losses.append(BCE.item())
            else:
                # Standard Autoencoder
                recon_batch = model(data)
                loss = F.binary_cross_entropy(recon_batch, data.view(-1, 784), reduction='mean')
                reconstruction_losses.append(loss.item())
            
            # MSE hesaplama
            mse = F.mse_loss(recon_batch, data.view(-1, 784), reduction='mean')
            mse_losses.append(mse.item())
            
            total_loss += loss.item() * batch_size
            total_mse += mse.item() * batch_size
            total_samples += batch_size
    
    avg_loss = total_loss / total_samples
    avg_mse = total_mse / total_samples
    
    print(f"\n=== {model_name} MODEL EVALUATION ===")
    print(f"Average Loss: {avg_loss:.6f}")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Std Reconstruction Loss: {np.std(reconstruction_losses):.6f}")
    print(f"Std MSE: {np.std(mse_losses):.6f}")
    
    return {
        'avg_loss': avg_loss,
        'avg_mse': avg_mse,
        'std_recon': np.std(reconstruction_losses),
        'std_mse': np.std(mse_losses)
    }

def visualize_latent_space_2d(model, test_loader, device, save_path='latent_space_2d.png'):
    """2D latent space görselleştirmesi (PCA ile boyut indirgeme)"""
    model.eval()
    
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(device)
            
            if hasattr(model, 'encode'):  # VAE
                mu, _ = model.encode(data.view(-1, 784))
                latent_vectors.append(mu.cpu().numpy())
            else:  # Standard AE
                encoded = model.encoder(data.view(-1, 784))
                latent_vectors.append(encoded.cpu().numpy())
            
            labels.append(label.numpy())
    
    latent_vectors = np.concatenate(latent_vectors)
    labels = np.concatenate(labels)
    
    # PCA ile 2D'ye indirge
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_vectors)
    
    # Görselleştirme
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title('Latent Space Visualization (PCA)')
    plt.xlabel(f'PC1 (Explained Variance: {pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'PC2 (Explained Variance: {pca.explained_variance_ratio_[1]:.3f})')
    
    # Her sınıf için merkez noktaları
    for i in range(10):
        mask = labels == i
        if np.any(mask):
            center = np.mean(latent_2d[mask], axis=0)
            plt.annotate(str(i), center, fontsize=12, fontweight='bold', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return pca.explained_variance_ratio_

def generate_interpolation(model, test_loader, device, n_steps=10):
    """İki latent nokta arasında interpolasyon"""
    model.eval()
    
    with torch.no_grad():
        # İki rastgele sample al
        data, _ = next(iter(test_loader))
        data = data.to(device)
        
        if hasattr(model, 'encode'):  # VAE
            mu1, _ = model.encode(data[0:1].view(-1, 784))
            mu2, _ = model.encode(data[1:2].view(-1, 784))
        else:  # Standard AE
            mu1 = model.encoder(data[0:1].view(-1, 784))
            mu2 = model.encoder(data[1:2].view(-1, 784))
        
        # Interpolasyon
        alphas = torch.linspace(0, 1, n_steps).to(device)
        interpolated_images = []
        
        for alpha in alphas:
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            
            if hasattr(model, 'decode'):  # VAE
                img_interp = model.decode(z_interp)
            else:  # Standard AE
                img_interp = model.decoder(z_interp)
            
            interpolated_images.append(img_interp.cpu().view(28, 28))
    
    # Görselleştirme
    fig, axes = plt.subplots(2, n_steps, figsize=(20, 4))
    
    # İlk satır: orijinal görüntüler
    axes[0][0].imshow(data[0].cpu().squeeze(), cmap='gray')
    axes[0][0].set_title('Start Image')
    axes[0][0].axis('off')
    
    axes[0][-1].imshow(data[1].cpu().squeeze(), cmap='gray')
    axes[0][-1].set_title('End Image')
    axes[0][-1].axis('off')
    
    for i in range(1, n_steps-1):
        axes[0][i].axis('off')
    
    # İkinci satır: interpolasyon
    for i, img in enumerate(interpolated_images):
        axes[1][i].imshow(img, cmap='gray')
        axes[1][i].set_title(f'α={alphas[i]:.1f}')
        axes[1][i].axis('off')
    
    plt.suptitle('Latent Space Interpolation', fontsize=16)
    plt.tight_layout()
    plt.savefig('interpolation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_reconstruction_quality(model, test_loader, device, num_samples=10):
    """Yeniden yapılandırma kalitesi analizi"""
    model.eval()
    
    with torch.no_grad():
        # Test verisinden örnekler al
        data, labels = next(iter(test_loader))
        data = data.to(device)
        
        if hasattr(model, 'forward') and len(model.forward(data[:1])) == 3:  # VAE
            recon_batch, _, _ = model(data)
        else:  # Standard AE
            recon_batch = model(data)
        
        # MSE hesaplama her örnek için
        mse_per_sample = []
        for i in range(min(num_samples, data.size(0))):
            mse = F.mse_loss(recon_batch[i], data[i].view(-1), reduction='mean')
            mse_per_sample.append(mse.item())
        
        # SSIM hesaplama (basit yaklaşım)
        ssim_scores = []
        for i in range(min(num_samples, data.size(0))):
            # Basit SSIM benzeri metric
            orig = data[i].view(28, 28).cpu().numpy()
            recon = recon_batch[i].view(28, 28).cpu().numpy()
            
            # Normalize
            orig = (orig - orig.mean()) / orig.std()
            recon = (recon - recon.mean()) / recon.std()
            
            # Correlation coefficient
            correlation = np.corrcoef(orig.flatten(), recon.flatten())[0, 1]
            ssim_scores.append(correlation)
    
    # Sonuçları görselleştir
    fig, axes = plt.subplots(3, num_samples, figsize=(20, 6))
    
    for i in range(num_samples):
        # Orijinal
        axes[0][i].imshow(data[i].cpu().squeeze(), cmap='gray')
        axes[0][i].set_title(f'Original\nLabel: {labels[i].item()}')
        axes[0][i].axis('off')
        
        # Yeniden yapılandırılmış
        axes[1][i].imshow(recon_batch[i].cpu().view(28, 28), cmap='gray')
        axes[1][i].set_title(f'Reconstructed\nMSE: {mse_per_sample[i]:.4f}')
        axes[1][i].axis('off')
        
        # Fark haritası
        diff = torch.abs(data[i] - recon_batch[i].view(28, 28)).cpu()
        axes[2][i].imshow(diff, cmap='hot')
        axes[2][i].set_title(f'Difference\nSSIM: {ssim_scores[i]:.3f}')
        axes[2][i].axis('off')
    
    plt.suptitle('Reconstruction Quality Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig('reconstruction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'avg_mse': np.mean(mse_per_sample),
        'avg_ssim': np.mean(ssim_scores),
        'mse_std': np.std(mse_per_sample),
        'ssim_std': np.std(ssim_scores)
    }

def comprehensive_evaluation():
    """Kapsamlı model değerlendirmesi"""
    
    # Veri yükleme
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST('./MNIST', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Modelleri yükle (varsayım: eğitilmiş modeller mevcut)
    from vae_implementation import VAE
    from comparison_autoencoder import StandardAutoencoder
    
    vae_model = VAE(latent_dim=20).to(device)
    std_ae_model = StandardAutoencoder(latent_dim=20).to(device)
    
    # Model ağırlıklarını yükle (eğer mevcut ise)
    try:
        vae_model.load_state_dict(torch.load('vae_model.pth', map_location=device))
        print("VAE model loaded successfully!")
    except:
        print("VAE model not found, using random weights for demonstration")
    
    try:
        std_ae_model.load_state_dict(torch.load('std_ae_model.pth', map_location=device))
        print("Standard AE model loaded successfully!")
    except:
        print("Standard AE model not found, using random weights for demonstration")
    
    # Değerlendirmeler
    print("=== COMPREHENSIVE MODEL EVALUATION ===")
    
    # 1. Temel metrikler
    vae_metrics = evaluate_model_quality(vae_model, test_loader, device, "VAE")
    std_ae_metrics = evaluate_model_quality(std_ae_model, test_loader, device, "Standard AE")
    
    # 2. Latent space analizi
    print("\n=== LATENT SPACE ANALYSIS ===")
    vae_variance = visualize_latent_space_2d(vae_model, test_loader, device, 'vae_latent_space.png')
    std_ae_variance = visualize_latent_space_2d(std_ae_model, test_loader, device, 'std_ae_latent_space.png')
    
    print(f"VAE PCA Explained Variance: {vae_variance[:2]}")
    print(f"Standard AE PCA Explained Variance: {std_ae_variance[:2]}")
    
    # 3. Interpolasyon analizi
    print("\n=== INTERPOLATION ANALYSIS ===")
    generate_interpolation(vae_model, test_loader, device)
    
    # 4. Yeniden yapılandırma kalitesi
    print("\n=== RECONSTRUCTION QUALITY ANALYSIS ===")
    vae_recon_quality = analyze_reconstruction_quality(vae_model, test_loader, device)
    std_ae_recon_quality = analyze_reconstruction_quality(std_ae_model, test_loader, device)
    
    # 5. Sonuç tablosu
    print("\n=== FINAL COMPARISON TABLE ===")
    print(f"{'Metric':<25} {'VAE':<15} {'Standard AE':<15}")
    print("-" * 55)
    print(f"{'Average Loss':<25} {vae_metrics['avg_loss']:<15.6f} {std_ae_metrics['avg_loss']:<15.6f}")
    print(f"{'Average MSE':<25} {vae_metrics['avg_mse']:<15.6f} {std_ae_metrics['avg_mse']:<15.6f}")
    print(f"{'Reconstruction MSE':<25} {vae_recon_quality['avg_mse']:<15.6f} {std_ae_recon_quality['avg_mse']:<15.6f}")
    print(f"{'SSIM Score':<25} {vae_recon_quality['avg_ssim']:<15.6f} {std_ae_recon_quality['avg_ssim']:<15.6f}")
    print(f"{'PCA Variance (PC1)':<25} {vae_variance[0]:<15.6f} {std_ae_variance[0]:<15.6f}")
    print(f"{'PCA Variance (PC2)':<25} {vae_variance[1]:<15.6f} {std_ae_variance[1]:<15.6f}")

if __name__ == "__main__":
    comprehensive_evaluation()
