import torch
import timm
from opacus.grad_sample import GradSampleModule

class AdaptiveFeatureExtractor(torch.nn.Module):
    def __init__(self, model_name='dinov2_base', num_classes=10):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Make top layer trainable for AFE
        self.trainable_head = torch.nn.Linear(self.backbone.embed_dim, self.backbone.embed_dim)
        self.trainable_head.train()
        
    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return self.trainable_head(features)
    
    def get_dp_module(self, config):
        """Wrap for DP-SGD with loose budget."""
        dp_model = GradSampleModule(self.trainable_head)
        return dp_model