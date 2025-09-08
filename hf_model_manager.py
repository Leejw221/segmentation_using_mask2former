import torch
import os
import logging
from typing import Optional, Dict, Any
from transformers import (
    Mask2FormerImageProcessor, 
    Mask2FormerForUniversalSegmentation,
    AutoImageProcessor,
    AutoModelForUniversalSegmentation
)
import yaml

class HuggingFaceModelManager:
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.device = self._detect_device()
        self.processor = None
        self.model = None
        self.model_name = self.config['model']['name']
        
    def _load_config(self, config_path: Optional[str]) -> dict:
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'config', 'hf_model_config.yaml'
            )
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        return {
            'model': {
                'name': 'facebook/mask2former-swin-small-ade-semantic',
                'device': 'auto',
                'confidence_threshold': 0.25,
                'available_models': [
                    'facebook/mask2former-swin-tiny-coco-instance',
                    'facebook/mask2former-swin-small-coco-instance', 
                    'facebook/mask2former-swin-base-coco-instance',
                    'facebook/mask2former-swin-large-coco-instance',
                    'facebook/mask2former-swin-small-coco-panoptic',
                    'facebook/mask2former-swin-base-coco-panoptic',
                    'facebook/mask2former-swin-large-coco-panoptic',
                    'facebook/mask2former-swin-small-ade-semantic',
                    'facebook/mask2former-swin-base-ade-semantic',
                    'facebook/mask2former-swin-large-ade-semantic'
                ]
            }
        }
    
    def _detect_device(self) -> str:
        device_config = self.config['model']['device']
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                self.logger.info(f"🚀 GPU detected: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                self.logger.info("💻 No GPU detected, using CPU")
        else:
            device = device_config
            
        return device
    
    def _update_dataset_from_model_name(self):
        """모델명을 기반으로 데이터셋 자동 감지 및 업데이트"""
        model_name_lower = self.model_name.lower()
        
        if 'cityscapes' in model_name_lower or 'mit-b' in model_name_lower:
            self.config['model']['dataset'] = 'cityscapes'
            self.logger.info("🏙️ Dataset auto-detected: Cityscapes (driving scenes)")
        elif 'ade' in model_name_lower:
            self.config['model']['dataset'] = 'ade20k'
            self.logger.info("🏠 Dataset auto-detected: ADE20k (indoor/outdoor scenes)")
        elif 'coco' in model_name_lower:
            self.config['model']['dataset'] = 'coco'
            self.logger.info("🎯 Dataset auto-detected: COCO (general objects)")
        else:
            # 기본값 유지
            if 'dataset' not in self.config['model']:
                self.config['model']['dataset'] = 'coco'
            self.logger.info(f"📋 Using configured dataset: {self.config['model']['dataset']}")
    
    def load_model(self, model_name: Optional[str] = None) -> bool:
        if model_name:
            self.model_name = model_name
        
        # 모델명을 기반으로 데이터셋 자동 감지
        self._update_dataset_from_model_name()
            
        try:
            self.logger.info(f"🔄 Loading HuggingFace model: {self.model_name}")
            
            # HuggingFace Transformers를 사용한 모델 로드
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForUniversalSegmentation.from_pretrained(self.model_name)
            
            if self.device == 'cuda':
                self.model = self.model.to(self.device)
                
            self.model.eval()
            self.logger.info(f"✅ HuggingFace model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model {self.model_name}: {e}")
            
            # 대안 방법으로 시도
            try:
                self.logger.info("🔄 Trying alternative loading method...")
                self.processor = Mask2FormerImageProcessor.from_pretrained(self.model_name)
                self.model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_name)
                
                if self.device == 'cuda':
                    self.model = self.model.to(self.device)
                    
                self.model.eval()
                self.logger.info(f"✅ Alternative loading successful on {self.device}")
                return True
                
            except Exception as e2:
                self.logger.error(f"❌ Alternative loading also failed: {e2}")
                return False
    
    def get_available_models(self) -> list:
        return self.config['model'].get('available_models', [])
    
    def switch_model(self, model_name: str) -> bool:
        if model_name in self.get_available_models():
            return self.load_model(model_name)
        else:
            self.logger.error(f"Model {model_name} not in available models list")
            return False
    
    def is_model_loaded(self) -> bool:
        return self.model is not None and self.processor is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'device': self.device,
            'is_loaded': self.is_model_loaded(),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'model_type': 'HuggingFace Transformers',
            'framework': 'transformers'
        }