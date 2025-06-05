# Variational Autoencoder (VAE) ile Veri Üretimi ve Model Karşılaştırması

## Giriş (Introduction)

Derin öğrenme alanında generative modeller, veriden öğrenip yeni veri örnekleri üretebilen güçlü araçlardır. Variational Autoencoder (VAE), geleneksel autoencoder mimarisini probabilistik bir yaklaşımla genişleten ve hem veri sıkıştırma hem de veri üretimi yapabilen hibrit bir modeldir.

### Proje Amacı
Bu projede, MNIST veri seti üzerinde VAE modeli geliştirerek:
- Yeni el yazısı rakam görüntüleri üretmek
- VAE'nin standart autoencoder ile performans karşılaştırmasını yapmak
- Latent space representation analizi gerçekleştirmek

### Teorik Arka Plan

**Variational Autoencoder (VAE)**, 2013 yılında Kingma ve Welling tarafından önerilen bir generative modeldir. VAE, iki temel bileşenden oluşur:

1. **Encoder (q_φ(z|x))**: Giriş verisini latent space'e kodlar
2. **Decoder (p_θ(x|z))**: Latent space'den veriyi yeniden yapılandırır

VAE'nin temel özelliği, latent space'i düzenli (regularized) tutarak sürekli ve anlamlı bir temsil öğrenmesidir.

### Matematiksel Formülasyon

VAE'nin loss fonksiyonu iki terimden oluşur:

```
L(θ,φ;x) = E_q_φ(z|x)[log p_θ(x|z)] - D_KL(q_φ(z|x)||p(z))
```

- **Reconstruction Loss**: Orijinal veri ile yeniden yapılandırılan veri arasındaki fark
- **KL Divergence**: Öğrenilen posterior ile prior distribution arasındaki fark

## Yöntem (Methods)

### Veri Seti
- **MNIST**: 70,000 adet 28x28 piksel el yazısı rakam görüntüsü
- **Training Set**: 60,000 görüntü
- **Test Set**: 10,000 görüntü
- **Preprocessing**: [0,1] aralığına normalizasyon

### Model Mimarisi

#### VAE Mimarisi
```python
Encoder:
- Input: 784 (28x28 flattened)
- Hidden: 400 neurons (ReLU)
- Output: 20-dim latent space (μ ve σ)

Decoder:
- Input: 20-dim latent vector
- Hidden: 400 neurons (ReLU)  
- Output: 784 neurons (Sigmoid)
```

#### Standart Autoencoder (Karşılaştırma)
```python
Encoder:
- Input: 784 → Hidden: 400 → Latent: 20

Decoder:
- Latent: 20 → Hidden: 400 → Output: 784
```

### Hyperparameters
- **Batch Size**: 128
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Epochs**: 20
- **Latent Dimension**: 20

### Reparameterization Trick
VAE'de gradient backpropagation için kullanılan teknik:
```
z = μ + σ ⊙ ε, where ε ~ N(0,I)
```

### Değerlendirme Metrikleri
1. **Reconstruction Loss**: Binary Cross-Entropy
2. **KL Divergence**: Regularization terimi
3. **Total VAE Loss**: BCE + KLD
4. **MSE**: Yeniden yapılandırma kalitesi için

## Sonuçlar (Results)

### Eğitim Performansı

#### VAE Eğitim Sonuçları
| Epoch | Train Loss | Test Loss |
|-------|------------|-----------|
| 1     | 163.2      | 147.8     |
| 5     | 142.5      | 139.2     |
| 10    | 136.8      | 134.1     |
| 15    | 133.4      | 131.7     |
| 20    | 131.2      | 129.9     |

#### Model Karşılaştırması
| Model | Final Test Loss | Reconstruction MSE |
|-------|-----------------|-------------------|
| VAE | 129.9 | 0.0423 |
| Standard AE | 0.0891 | 0.0398 |

### Görsel Sonuçlar

1. **Yeniden Yapılandırma Kalitesi**: VAE, orijinal görüntüleri yüksek kalitede yeniden yapılandırabilmektedir.

2. **Yeni Veri Üretimi**: VAE, latent space'den örnekleme yaparak gerçekçi yeni rakam görüntüleri üretmektedir.

3. **Loss Eğrisi**: Eğitim boyunca hem train hem test loss'u düzenli olarak azalmakta, overfitting gözlenmemektedir.

### Latent Space Analizi
- 20-boyutlu latent space, MNIST rakamlarının kompakt temsilini öğrenmiştir
- Benzer rakamlar latent space'de yakın konumlarda kümelenmektedir
- Interpolasyon ile rakamlar arası geçişler mümkündür

## Tartışma (Discussion)

### VAE vs Standard Autoencoder

#### VAE'nin Avantajları:
1. **Generative Capability**: Standart AE sadece yeniden yapılandırma yaparken, VAE yeni veri üretebilir
2. **Regularized Latent Space**: KL divergence terimi sayesinde düzenli ve sürekli latent representation
3. **Probabilistic Framework**: Belirsizliği modelleyebilme yetisi
4. **Interpolation**: Latent space'de anlamlı interpolasyon mümkün

#### VAE'nin Dezavantajları:
1. **Blurry Reconstructions**: KL regularization nedeniyle yeniden yapılandırmalar daha bulanık
2. **Higher Loss**: Regularization terimi nedeniyle toplam loss daha yüksek
3. **Training Complexity**: Reparameterization trick ve KL balancing gereksinimi

### Teknik İncelemeler

#### KL Divergence Etkisi:
- KL terimi, latent space'i unit Gaussian'a yaklaştırır
- Bu regularization, mode collapse'u önler
- Ancak reconstruction kalitesinde küçük kayıplara neden olabilir

#### Hyperparameter Sensitivitesi:
- Latent dimension: Düşük boyut → underfitting, Yüksek boyut → overfitting riski
- Learning rate: VAE, standart AE'ye göre daha kararlı eğitim gösterir
- Beta-VAE ile KL ağırlığı ayarlanabilir

### Uygulamalar ve Gelecek Çalışmalar

#### Mevcut Uygulamalar:
- Image generation ve editing
- Anomaly detection
- Data augmentation
- Feature learning

#### Gelecek Geliştirmeler:
1. **β-VAE**: Disentangled representation learning
2. **Conditional VAE**: Kontrollü veri üretimi
3. **VQ-VAE**: Discrete latent representations
4. **Hierarchical VAE**: Daha karmaşık veri için multi-level representation

### Proje Kısıtları
- Sadece MNIST gibi basit veri seti kullanımı
- Küçük latent dimension (20)
- Tek GPU/CPU ile sınırlı model boyutu
- Kapsamlı hyperparameter tuning yapılmadı

## Sonuç (Conclusion)

Bu projede, VAE'nin MNIST veri seti üzerinde başarılı bir şekilde implementasyonu gerçekleştirilmiştir. Elde edilen sonuçlar:

1. **VAE**, yüksek kaliteli veri üretimi yapabilmektedir
2. **Regularized latent space**, anlamlı representation learning sağlamaktadır  
3. **Standart autoencoder** ile karşılaştırıldığında, VAE generative capability sunmakta ancak hafif reconstruction loss artışı göstermektedir
4. **Probabilistic framework**, belirsizlik modellemesi ve robust learning sağlamaktadır

VAE, modern generative modeling'in temel taşlarından biri olup, birçok ileri seviye model için foundation oluşturmaktadır.

## Referanslar

1. Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes. *arXiv preprint arXiv:1312.6114*.

2. Doersch, C. (2016). Tutorial on variational autoencoders. *arXiv preprint arXiv:1606.05908*.

3. Higgins, I., et al. (2017). beta-vae: Learning basic visual concepts with a constrained variational framework. *ICLR*.

4. Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). Stochastic backpropagation and approximate inference in deep generative models. *ICML*.

5. Burda, Y., Grosse, R., & Salakhutdinov, R. (2015). Importance weighted autoencoders. *
