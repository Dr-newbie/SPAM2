import os
from abc import abstractmethod
import traceback

import torch
import timm
import einops
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

try:
    # 모듈 실행 권장: python -m code.utils.run_contrastive
    from code.models.transforms_utils import get_eval_transforms, get_constants
except ModuleNotFoundError:
    # 동일 디렉토리(relative import) 폴백
    from .transforms_utils import get_eval_transforms, get_constants


class InferenceEncoder(torch.nn.Module):
    def __init__(self, weights_path = None, **build_kwargs):
        super(InferenceEncoder, self).__init__()

        self.weights_path = weights_path

        self.encoder , self.eval_transforms, self.precision = self._build(weights_path, **build_kwargs)

        self.out_embed_dim = self.get_output_dim()

    def forward_features(self,x):
        pass

    def forward(self, x , **kwargs):
        embedding = self.forward_features(x)
        return embedding
    
    def get_output_dim(self):
        """
        weights_path 유무와 무관하게 encoder 인스턴스의 표준 속성들로 출력 차원을 추론한다.
        """
        e = self.encoder
        # timm ViT/UNI/Virchow 계열: head_hidden_size
        if hasattr(e, "head_hidden_size"):
            return int(e.head_hidden_size)
        # HF ViT/CLIPVision 등: config.hidden_size
        if hasattr(e, "config") and hasattr(e.config, "hidden_size"):
            return int(e.config.hidden_size)
        # OpenCLIP/CONCH 등: embed_dim
        if hasattr(e, "embed_dim"):
            return int(e.embed_dim)
        # 일부 timm 모델: num_features
        if hasattr(e, "num_features"):
            return int(e.num_features)
        # CONCH/TITAN (attentional pooler): trunk.embed_dim 또는 attentional pooler의 output
        if hasattr(e, "trunk") and hasattr(e.trunk, "embed_dim"):
            # TITAN/CONCH의 경우 attentional pooler가 있으면 768로 변환
            if hasattr(e, "attentional_pooler"):
                return 768  # TITAN CONCH v1.5 pooled output dimension
            return int(e.trunk.embed_dim)
        # CONCH visual_encoder: visual_model의 embed_dim
        if hasattr(e, "visual_model") and hasattr(e.visual_model, "embed_dim"):
            return int(e.visual_model.embed_dim)
        # CONCH model_cfg에서 추론
        if hasattr(e, "model_cfg") and hasattr(e.model_cfg, "vision_cfg"):
            if hasattr(e.model_cfg.vision_cfg, "width"):
                return int(e.model_cfg.vision_cfg.width)
        raise ValueError(
            "Cannot infer output dim from encoder. "
            "Please add a new branch for this backbone."
        )

        
    @abstractmethod
    def _build(self, **build_kwargs):
        pass


class ConchInferenceEncoder(InferenceEncoder):
    
    def _build(self, weights_path):
        try:
            from conch.open_clip_custom import create_model_from_pretrained
        except:
            traceback.print_exc()
            raise Exception("Please install CONCH `pip install git+https://github.com/Mahmoodlab/CONCH.git`")
        
        try:
            if weights_path is None:
                model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch")
            else:
                model, preprocess = create_model_from_pretrained('conch_ViT-B-16', 
                                                                 checkpoint_path = os.path.join(weights_path, "pytorch_model.bin"))
        except:
            traceback.print_exc()
            raise Exception("Failed to download CONCH model, make sure that you were granted access and that you correctly registered your token")
        
        eval_transform = preprocess
        precision = torch.float32
        
        return model, eval_transform, precision
    
    def forward_features(self, x):
        embedding = self.encoder.encode_image(x, proj_contrast=False, normalize=False)
        return embedding


class PhikonInferenceEncoder(InferenceEncoder):
    def _build(self, weights_path):
        from transformers import ViTModel
        
        model = ViTModel.from_pretrained(weights_path, add_pooling_layer=False)
        mean, std = get_constants('imagenet')
        eval_transform = get_eval_transforms(mean, std)
        precision = torch.float32
        
        return model, eval_transform, precision
    
    def forward_features(self, x):
        out = self.encoder(pixel_values=x)
        out = out.last_hidden_state[:, 0, :]
    
        return out
    

class PlipInferenceEncoder(InferenceEncoder):
    def _build(self, weights_path):
        from transformers import CLIPImageProcessor, CLIPVisionModel

        if weights_path is None:
            weights_path = "vinid/plip"
        else:
            assert os.path.exists(weights_path), f"Invalid weights_path! >> {weights_path}"
        
        img_transforms_clip = CLIPImageProcessor.from_pretrained(weights_path)
        model = CLIPVisionModel.from_pretrained(weights_path)  # Use for feature extraction
        def _eval_transform(img): return img_transforms_clip(
            img, return_tensors='pt')['pixel_values'].squeeze(0)
        eval_transform = _eval_transform
        precision = torch.float32
        
        return model, eval_transform, precision
    
    def forward_features(self, x):
        embedding = self.encoder(x).pooler_output
        return embedding
     
    
class UNIInferenceEncoder(InferenceEncoder):
    def _build(
        self, 
        weights_path,
        timm_kwargs={"dynamic_img_size": True, "num_classes": 0, "init_values": 1.0}
    ):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        
        try:
            if weights_path is None:
                model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, **timm_kwargs)
            else:
                assert os.path.exists(weights_path), f"Invalid weights_path! >> {weights_path}"

                model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, **timm_kwargs)
                model.load_state_dict(torch.load(os.path.join(weights_path, "pytorch_model.bin"), weights_only=True), strict=True) # , map_location="cpu"                

        except:
            traceback.print_exc()
            raise Exception("Failed to download UNI model, make sure that you were granted access and that you correctly registered your token")
        
        eval_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        precision = torch.float16
        return model, eval_transform, precision
    
    def forward_features(self, x):
        embedding = self.encoder(x)
        return embedding
    

class GigaPathInferenceEncoder(InferenceEncoder):
    def _build(
        self, 
        weights_path,
        timm_kwargs={"num_classes": 0, "init_values": 1e-5}
        ):
        import timm
        from torchvision import transforms
        
        if weights_path is None:
            model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True, **timm_kwargs)
        else:
            assert os.path.exists(weights_path), f"Invalid weights_path! >> {weights_path}"

            model = timm.create_model("vit_giant_patch14_dinov2", img_size=224, patch_size=16, **timm_kwargs)
            model.load_state_dict(torch.load(os.path.join(weights_path, "pytorch_model.bin"), weights_only=True), strict=True)

        eval_transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        precision = torch.float32
        return model, eval_transform, precision
    
    def forward_features(self, x):
        embedding = self.encoder(x)
        return embedding
    

class VirchowInferenceEncoder(InferenceEncoder):

    def _build(
        self,
        weights_path,
        return_cls=False,
        timm_kwargs={'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU}
    ):
        if weights_path is None:
            model = timm.create_model(
                "hf-hub:paige-ai/Virchow",
                pretrained=True,
                **timm_kwargs
            )           
        else:
            assert os.path.exists(weights_path), f"Invalid weights_path! >> {weights_path}"

            model = timm.create_model("vit_huge_patch14_224", pretrained=False,
                                      checkpoint_path=os.path.join(weights_path, "pytorch_model.bin"),
                                      img_size=224, init_values=1e-5, num_classes=0, mlp_ratio=5.3375,
                                      global_pool="",dynamic_img_size=True,
                                      **timm_kwargs)

        eval_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

        precision = torch.float32
        self.return_cls = return_cls
        if not self.return_cls:
            model.head_hidden_size *=2 if not self.return_cls else model.head_hidden_size # double the hidden size if cls token is not returned
    
        return model, eval_transform, precision

    def forward_features(self, x):
        output = self.encoder(x)
    
        class_token = output[:, 0]
        if self.return_cls:
            embedding = class_token            
        else:
            patch_tokens = output[:, 1:]
            embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)

        return embedding


class Virchow2InferenceEncoder(InferenceEncoder):
    import timm
    
    def _build(
        self,
        weights_path,
        return_cls=False,
        timm_kwargs={'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU}
    ):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        if weights_path is None:
            model = timm.create_model(
                "hf-hub:paige-ai/Virchow2",
                pretrained=True,
                **timm_kwargs
            )            
        else:
            assert os.path.exists(weights_path), f"Invalid weights_path! >> {weights_path}"

            model = timm.create_model("vit_huge_patch14_224", pretrained=False,
                                      checkpoint_path=os.path.join(weights_path, "pytorch_model.bin"),
                                      img_size=224, init_values=1e-5, num_classes=0, mlp_ratio=5.3375, reg_tokens=4, # reg_tokens=4 is for Virchow2
                                      global_pool="",dynamic_img_size=True,
                                      **timm_kwargs)
        
        eval_transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        precision = torch.float16
        self.return_cls = return_cls
        if not self.return_cls:
            model.head_hidden_size *=2 if not self.return_cls else model.head_hidden_size # double the hidden size if cls token is not returned
        
        return model, eval_transform, precision

    def forward_features(self, x):
        output = self.encoder(x)
    
        class_token = output[:, 0]
        if self.return_cls:
            embedding = class_token            
        else:
            patch_tokens = output[:, 5:]
            embedding = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)

        return embedding



class HOptimus0InferenceEncoder(InferenceEncoder):
    
    def _build(
        self,
        weights_path,
        timm_kwargs={'init_values': 1e-5, 'dynamic_img_size': False}
    ):
        import timm
        from torchvision import transforms

        if weights_path is None:
            model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, **timm_kwargs)
        else:
            assert os.path.exists(weights_path), f"Invalid weights_path! >> {weights_path}"

            model = timm.create_model("vit_giant_patch14_reg4_dinov2", pretrained=False, 
                                      checkpoint_path=os.path.join(weights_path, "pytorch_model.bin"), 
                                      img_size=224, 
                                      global_pool="token", 
                                      num_classes=0,
                                      **timm_kwargs)

        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.707223, 0.578729, 0.703617), 
                std=(0.211883, 0.230117, 0.177517)
            ),
        ])
        
        precision = torch.float16
        return model, eval_transform, precision
    def forward_features(self, x):
        embedding = self.encoder(x)
        return embedding

class TiTanEncoder(InferenceEncoder):
    
    def _build(
        self,
        weights_path,
        timm_kwargs={'init_values': 1e-5, 'dynamic_img_size': False}
    ):
        from transformers import AutoModel

        # TiTAN provides a CONCH v1.5 patch encoder and eval transform via return_conch().
        model_id = weights_path or "MahmoodLab/TITAN"
        titan = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        model, eval_transform = titan.return_conch()
        
        # TITAN/CONCH는 CUDA에서 einsum 문제가 있으므로 CPU에서 실행
        # 모델을 CPU에 고정하고 float32 사용
        model = model.cpu()
        for param in model.parameters():
            param.data = param.data.cpu()
        precision = torch.float32
        return model, eval_transform, precision
    
    def forward_features(self, x):
        # CONCH는 CPU에서만 안정적이므로 CPU로 이동하여 처리
        original_device = x.device
        x_cpu = x.cpu().float()  # CPU로 이동하고 float32 변환
        
        # 모델의 모든 파라미터가 CPU에 있는지 확인
        self.encoder = self.encoder.cpu()
        
        embedding = self.encoder(x_cpu)
        
        # 결과를 원래 device로 되돌림
        embedding = embedding.to(original_device)
        return embedding


def inf_encoder_factory(enc_name):
    if enc_name == 'conch_v1':
        return ConchInferenceEncoder
    elif enc_name == 'uni_v1': # 이건 token access 됨
        return UNIInferenceEncoder
    elif enc_name == 'phikon':
        return PhikonInferenceEncoder
    elif enc_name == 'plip':
        return PlipInferenceEncoder
    elif enc_name == 'gigapath':
        return GigaPathInferenceEncoder
    elif enc_name == 'virchow': # 이거 token access 안됨
        return VirchowInferenceEncoder
        # return VirchowTokenizerEncoder
    elif enc_name == 'virchow2': # 이건 token access 됨
        return Virchow2InferenceEncoder
    elif enc_name == 'hoptimus0': # 이건 token access 됨
        return HOptimus0InferenceEncoder
    elif enc_name == 'titan': # token access 걸어둠
        return TiTanEncoder
    else:
        raise ValueError(f"Unknown encoder name {enc_name}")
    
