import torch, torch.nn as nn
import numpy as np
 
EMOTIONS = ['neutral','calm','happy','sad','angry','fearful','disgust','surprised']
 
class SpeechEmotionCNN(nn.Module):
    def __init__(self, n_mfcc=40, n_cls=8):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1,32,(3,3),padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,(3,3),padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
       nn.Conv2d(64,128,(3,3),padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4)))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4),
            nn.Linear(128*4*4, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, n_cls))
    def forward(self, x): return self.classifier(self.cnn(x))
 
def extract_features(waveform, sr=22050, n_mfcc=40, max_len=128):
    try:
        import librosa
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)
        delta = librosa.feature.delta(mfcc)
        features = np.vstack([mfcc, delta])
    except ImportError:
        features = np.random.rand(2*n_mfcc, max_len)
    if features.shape[1] < max_len:
        features = np.pad(features, ((0,0),(0,max_len-features.shape[1])))
    return features[:, :max_len]
 
def augment(features):
    """Time masking and frequency masking (SpecAugment)."""
    f = features.copy()
    t_mask = np.random.randint(0, features.shape[1]//4)
    t_start = np.random.randint(0, features.shape[1]-t_mask)
    f[:, t_start:t_start+t_mask] = 0
    f_mask = np.random.randint(0, features.shape[0]//4)
    f_start = np.random.randint(0, features.shape[0]-f_mask)
    f[f_start:f_start+f_mask, :] = 0
    return f
 
model = SpeechEmotionCNN()
dummy = np.random.rand(22050)
feats = extract_features(dummy)
aug = augment(feats)
x = torch.FloatTensor(feats).unsqueeze(0).unsqueeze(0)
out = model(x)
print(f"Feature shape: {feats.shape} | Model output: {out.shape}")
print(f"Predicted emotion: {EMOTIONS[out.argmax().item()]}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
