import torch
import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Tuple, Optional
import logging
from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation

class HuggingFaceSegmentationProcessor:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.logger = logging.getLogger(__name__)
        
        # 데이터셋에 따른 클래스 설정
        dataset = self.model_manager.config['model'].get('dataset', 'coco')
        self.dataset = dataset
        
        if dataset == 'cityscapes':
            self.classes = self._get_cityscapes_classes()
        elif dataset == 'ade20k':
            self.classes = self._get_ade20k_classes()
        elif dataset == 'mapillary-vistas':
            self.classes = self._get_mapillary_vistas_classes()
        else:  # 기본값은 COCO-Stuff
            self.classes = self._get_coco_stuff_classes()
        
        # 하위 호환성을 위해 coco_classes도 유지
        self.coco_classes = self.classes
        
    def _get_coco_stuff_classes(self) -> Dict[int, str]:
        """COCO-Stuff 데이터셋 클래스 (공식 183개 클래스: things 0-79 + stuff 92-182)"""
        return {
            # Background
            0: 'unlabeled',
            
            # Things Classes (1-80) - COCO 공식 80개 객체 클래스
            1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
            6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
            11: 'fire hydrant', 12: 'stop sign', 13: 'parking meter', 14: 'bench',
            15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep', 20: 'cow',
            21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe', 25: 'backpack',
            26: 'umbrella', 27: 'handbag', 28: 'tie', 29: 'suitcase', 30: 'frisbee',
            31: 'skis', 32: 'snowboard', 33: 'sports ball', 34: 'kite', 35: 'baseball bat',
            36: 'baseball glove', 37: 'skateboard', 38: 'surfboard', 39: 'tennis racket',
            40: 'bottle', 41: 'wine glass', 42: 'cup', 43: 'fork', 44: 'knife',
            45: 'spoon', 46: 'bowl', 47: 'banana', 48: 'apple', 49: 'sandwich',
            50: 'orange', 51: 'broccoli', 52: 'carrot', 53: 'hot dog', 54: 'pizza',
            55: 'donut', 56: 'cake', 57: 'chair', 58: 'couch', 59: 'potted plant',
            60: 'bed', 61: 'dining table', 62: 'toilet', 63: 'tv', 64: 'laptop',
            65: 'mouse', 66: 'remote', 67: 'keyboard', 68: 'cell phone', 69: 'microwave',
            70: 'oven', 71: 'toaster', 72: 'sink', 73: 'refrigerator', 74: 'book',
            75: 'clock', 76: 'vase', 77: 'scissors', 78: 'teddy bear', 79: 'hair drier',
            80: 'toothbrush',
            
            # Stuff Classes (92-182) - COCO-Stuff 공식 91개 stuff 클래스
            92: 'banner', 93: 'blanket', 94: 'branch', 95: 'bridge', 96: 'building-other',
            97: 'bush', 98: 'cabinet', 99: 'cage', 100: 'cardboard', 101: 'carpet',
            102: 'ceiling-other', 103: 'ceiling-tile', 104: 'cloth', 105: 'clothes',
            106: 'clouds', 107: 'counter', 108: 'cupboard', 109: 'curtain', 110: 'desk-stuff',
            111: 'dirt', 112: 'door-stuff', 113: 'fence', 114: 'floor-marble', 115: 'floor-other',
            116: 'floor-stone', 117: 'floor-tile', 118: 'floor-wood', 119: 'flower', 120: 'fog',
            121: 'food-other', 122: 'fruit', 123: 'furniture-other', 124: 'grass', 125: 'gravel',
            126: 'ground-other', 127: 'hill', 128: 'house', 129: 'leaves', 130: 'light',
            131: 'mat', 132: 'metal', 133: 'mirror-stuff', 134: 'moss', 135: 'mountain',
            136: 'mud', 137: 'napkin', 138: 'net', 139: 'paper', 140: 'pavement',
            141: 'pillow', 142: 'plant-other', 143: 'plastic', 144: 'platform', 145: 'playingfield',
            146: 'railing', 147: 'railroad', 148: 'river', 149: 'road', 150: 'rock',
            151: 'roof', 152: 'rug', 153: 'salad', 154: 'sand', 155: 'sea',
            156: 'shelf', 157: 'sky-other', 158: 'skyscraper', 159: 'snow', 160: 'solid-other',
            161: 'stairs', 162: 'stone', 163: 'straw', 164: 'structural-other', 165: 'table',
            166: 'tent', 167: 'textile-other', 168: 'towel', 169: 'tree', 170: 'vegetable',
            171: 'wall-brick', 172: 'wall-concrete', 173: 'wall-other', 174: 'wall-panel',
            175: 'wall-stone', 176: 'wall-tile', 177: 'wall-wood', 178: 'water-other',
            179: 'waterdrops', 180: 'window-blind', 181: 'window-other', 182: 'wood'
        }
    
    def _get_ade20k_classes(self) -> Dict[int, str]:
        """ADE20K 데이터셋 클래스 정의 (SceneParse150 - 150개 클래스)"""
        return {
            # 주요 150개 클래스 - 전체 리스트를 위해 공식 문서 참조 필요
            0: 'background',
            1: 'wall', 2: 'building', 3: 'sky', 4: 'floor', 5: 'tree',
            6: 'ceiling', 7: 'road', 8: 'bed', 9: 'windowpane', 10: 'grass',
            11: 'cabinet', 12: 'sidewalk', 13: 'person', 14: 'earth', 15: 'door',
            16: 'table', 17: 'mountain', 18: 'plant', 19: 'curtain', 20: 'chair',
            21: 'car', 22: 'water', 23: 'painting', 24: 'sofa', 25: 'shelf',
            26: 'house', 27: 'sea', 28: 'mirror', 29: 'rug', 30: 'field',
            31: 'armchair', 32: 'seat', 33: 'fence', 34: 'desk', 35: 'rock',
            36: 'wardrobe', 37: 'lamp', 38: 'bathtub', 39: 'railing', 40: 'cushion',
            41: 'base', 42: 'box', 43: 'column', 44: 'signboard', 45: 'chest of drawers',
            46: 'counter', 47: 'sand', 48: 'sink', 49: 'skyscraper', 50: 'fireplace',
            51: 'refrigerator', 52: 'grandstand', 53: 'path', 54: 'stairs', 55: 'runway',
            56: 'case', 57: 'pool table', 58: 'pillow', 59: 'screen door', 60: 'stairway',
            61: 'river', 62: 'bridge', 63: 'bookcase', 64: 'blind', 65: 'coffee table',
            66: 'toilet', 67: 'flower', 68: 'book', 69: 'hill', 70: 'bench',
            71: 'countertop', 72: 'stove', 73: 'palm', 74: 'kitchen island', 75: 'computer',
            76: 'swivel chair', 77: 'boat', 78: 'bar', 79: 'arcade machine', 80: 'hovel',
            81: 'bus', 82: 'towel', 83: 'light', 84: 'truck', 85: 'tower',
            86: 'chandelier', 87: 'awning', 88: 'streetlight', 89: 'booth', 90: 'television receiver',
            91: 'airplane', 92: 'dirt track', 93: 'apparel', 94: 'pole', 95: 'land',
            96: 'bannister', 97: 'escalator', 98: 'ottoman', 99: 'bottle', 100: 'buffet',
            101: 'poster', 102: 'stage', 103: 'van', 104: 'ship', 105: 'fountain',
            106: 'conveyer belt', 107: 'canopy', 108: 'washer', 109: 'plaything', 110: 'swimming pool',
            111: 'stool', 112: 'barrel', 113: 'basket', 114: 'waterfall', 115: 'tent',
            116: 'bag', 117: 'minibike', 118: 'cradle', 119: 'oven', 120: 'ball',
            121: 'food', 122: 'step', 123: 'tank', 124: 'trade name', 125: 'microwave',
            126: 'pot', 127: 'animal', 128: 'bicycle', 129: 'lake', 130: 'dishwasher',
            131: 'screen', 132: 'blanket', 133: 'sculpture', 134: 'hood', 135: 'sconce',
            136: 'vase', 137: 'traffic light', 138: 'tray', 139: 'ashcan', 140: 'fan',
            141: 'pier', 142: 'crt screen', 143: 'plate', 144: 'monitor', 145: 'bulletin board',
            146: 'shower', 147: 'radiator', 148: 'glass', 149: 'clock'
            # 마지막 클래스는 149번으로 총 150개 (0-149)
        }
    
    def _get_cityscapes_classes(self) -> Dict[int, str]:
        """Cityscapes 데이터셋 클래스 정의 - HuggingFace 공식 19개 클래스"""
        return {
            0: 'road',
            1: 'sidewalk', 
            2: 'building',
            3: 'wall',
            4: 'fence',
            5: 'pole',
            6: 'traffic light',
            7: 'traffic sign',
            8: 'vegetation',
            9: 'terrain',
            10: 'sky',
            11: 'person',
            12: 'rider',
            13: 'car',
            14: 'truck',
            15: 'bus',
            16: 'train',
            17: 'motorcycle',
            18: 'bicycle'
        }
    
    def _get_mapillary_vistas_classes(self) -> Dict[int, str]:
        """Mapillary Vistas 데이터셋 클래스 정의 - 공식 논문 66개 클래스 (v1.2)"""
        return {
            # Animals
            0: 'animal--bird',
            1: 'animal--ground-animal',
            
            # Construction - Barriers
            2: 'construction--barrier--curb',
            3: 'construction--barrier--fence',
            4: 'construction--barrier--guard-rail',
            5: 'construction--barrier--other-barrier',
            6: 'construction--barrier--wall',
            
            # Construction - Flat surfaces
            7: 'construction--flat--bike-lane',
            8: 'construction--flat--crosswalk-plain',
            9: 'construction--flat--curb-cut',
            10: 'construction--flat--driveway',
            11: 'construction--flat--parking',
            12: 'construction--flat--pedestrian-area',
            13: 'construction--flat--rail-track',
            14: 'construction--flat--road',
            15: 'construction--flat--service-lane',
            16: 'construction--flat--sidewalk',
            17: 'construction--flat--traffic-island',
            
            # Construction - Structures
            18: 'construction--structure--bridge',
            19: 'construction--structure--building',
            20: 'construction--structure--tunnel',
            
            # Humans
            21: 'human--person',
            22: 'human--rider',
            
            # Markings
            23: 'marking--crosswalk-zebra',
            24: 'marking--general',
            
            # Nature
            25: 'nature--mountain',
            26: 'nature--sand',
            27: 'nature--sky',
            28: 'nature--snow',
            29: 'nature--terrain',
            30: 'nature--vegetation',
            31: 'nature--water',
            
            # Objects
            32: 'object--banner',
            33: 'object--bench',
            34: 'object--bike-rack',
            35: 'object--catch-basin',
            36: 'object--cctv-camera',
            37: 'object--fire-hydrant',
            38: 'object--junction-box',
            39: 'object--mailbox',
            40: 'object--manhole',
            41: 'object--phone-booth',
            42: 'object--pothole',
            43: 'object--street-light',
            
            # Object - Support structures
            44: 'object--support--pole',
            45: 'object--support--pole-group',
            46: 'object--support--traffic-sign-frame',
            47: 'object--support--utility-pole',
            
            # Object - Traffic elements
            48: 'object--traffic-light',
            49: 'object--traffic-sign--back',
            50: 'object--traffic-sign--front',
            51: 'object--trash-can',
            
            # Object - Vehicles
            52: 'object--vehicle--bicycle',
            53: 'object--vehicle--boat',
            54: 'object--vehicle--bus',
            55: 'object--vehicle--car',
            56: 'object--vehicle--caravan',
            57: 'object--vehicle--motorcycle',
            58: 'object--vehicle--on-rails',
            59: 'object--vehicle--other-vehicle',
            60: 'object--vehicle--trailer',
            61: 'object--vehicle--truck',
            62: 'object--vehicle--wheeled-slow',
            
            # Void classes
            63: 'void--ground',
            64: 'void--static',
            65: 'void--unlabeled'
        }
    
    def process_image(self, image: np.ndarray) -> Tuple[Dict, np.ndarray]:
        if not self.model_manager.is_model_loaded():
            self.logger.error("Model not loaded")
            return {}, image
        
        try:
            # OpenCV BGR -> RGB 변환
            if len(image.shape) == 3 and image.shape[2] == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # 이미지 전처리
            inputs = self.model_manager.processor(images=pil_image, return_tensors="pt")
            
            # GPU로 이동 (필요시)
            if self.model_manager.device == 'cuda':
                inputs = {k: v.to(self.model_manager.device) for k, v in inputs.items()}
            
            # 모델 추론
            with torch.no_grad():
                outputs = self.model_manager.model(**inputs)
            
            # 모델 타입에 따른 후처리
            model_name = self.model_manager.model_name
            if 'semantic' in model_name:
                # Semantic segmentation 후처리
                results = self.model_manager.processor.post_process_semantic_segmentation(
                    outputs,
                    target_sizes=[pil_image.size[::-1]]
                )[0]
                # semantic 결과를 instance 형식으로 변환
                results = self._convert_semantic_to_instance_format(results)
            elif 'panoptic' in model_name:
                # Panoptic segmentation 후처리
                results = self.model_manager.processor.post_process_panoptic_segmentation(
                    outputs,
                    target_sizes=[pil_image.size[::-1]],
                    threshold=self.model_manager.config['model']['confidence_threshold']
                )[0]
            else:
                # Instance segmentation 후처리 (기존 방식)
                results = self.model_manager.processor.post_process_instance_segmentation(
                    outputs, 
                    target_sizes=[pil_image.size[::-1]],
                    threshold=self.model_manager.config['model']['confidence_threshold']
                )[0]
            
            # 시각화 생성
            visualization = self._create_visualization(image, results)
            
            # 디버깅: 감지된 클래스 ID들 출력 (개발시에만 사용)
            # labels = results.get('labels', [])
            # if labels:
            #     detected_classes = []
            #     for label in labels:
            #         label_id = label.item() if hasattr(label, 'item') else label
            #         class_name = self.classes.get(label_id, f'class_{label_id}')
            #         detected_classes.append(f"ID:{label_id}({class_name})")
            #     self.logger.info(f"🔍 Detected classes: {', '.join(detected_classes)}")
            
            # 결과 정리
            processed_results = {
                'masks': results.get('segments_info', []) if 'segments_info' in results else results.get('masks', []),
                'labels': results.get('labels', []),
                'scores': results.get('scores', []),
                'class_names': [self.classes.get(label.item() if hasattr(label, 'item') else label, f'class_{label}') 
                               for label in results.get('labels', [])]
            }
            
            return processed_results, visualization
            
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {}, image
    
    def _create_visualization(self, original_image: np.ndarray, results: Dict) -> np.ndarray:
        vis_image = original_image.copy()
        
        # 마스크 정보 추출 (다양한 결과 형식 지원)
        masks = None
        labels = results.get('labels', [])
        scores = results.get('scores', [])
        
        if 'masks' in results and len(results['masks']) > 0:
            masks = results['masks']
        elif 'segments_info' in results and len(results['segments_info']) > 0:
            # Panoptic segmentation의 경우
            segmentation = results.get('segmentation', None)
            if segmentation is not None:
                return self._create_panoptic_visualization(original_image, results)
        
        if masks is None or len(masks) == 0:
            return vis_image
        
        # 각 마스크에 대해 시각화
        for i, mask in enumerate(masks):
            try:
                # 마스크를 numpy array로 변환
                if torch.is_tensor(mask):
                    mask_np = mask.cpu().numpy().astype(np.uint8)
                else:
                    mask_np = mask.astype(np.uint8)
                
                # 마스크가 2D가 아닌 경우 처리
                if len(mask_np.shape) > 2:
                    mask_np = mask_np.squeeze()
                
                # 클래스 ID에 따른 고정 색상 사용
                if i < len(labels):
                    label_id = labels[i].item() if torch.is_tensor(labels[i]) else labels[i]
                    color = self._get_class_color(label_id)
                else:
                    color = self._get_class_color(i)  # 기본값
                
                colored_mask = np.zeros_like(vis_image)
                colored_mask[mask_np > 0] = color
                
                # 마스크 오버레이 (더 선명하게)
                vis_image = cv2.addWeighted(vis_image, 0.4, colored_mask, 0.6, 0)
                
            except Exception as e:
                self.logger.warning(f"Error visualizing mask {i}: {e}")
                continue
        
        return vis_image
    
    def _create_panoptic_visualization(self, original_image: np.ndarray, results: Dict) -> np.ndarray:
        vis_image = original_image.copy()
        
        segmentation = results.get('segmentation', None)
        segments_info = results.get('segments_info', [])
        
        if segmentation is None:
            return vis_image
        
        # segmentation을 numpy array로 변환
        if torch.is_tensor(segmentation):
            segmentation = segmentation.cpu().numpy()
        
        for i, segment in enumerate(segments_info):
            segment_id = segment.get('id', i)
            label_id = segment.get('label_id', 0)
            
            # 해당 세그먼트의 마스크 생성
            mask = (segmentation == segment_id).astype(np.uint8)
            
            if mask.sum() > 0:
                # 클래스 ID에 따른 고정 색상 사용
                color = self._get_class_color(label_id)
                colored_mask = np.zeros_like(vis_image)
                colored_mask[mask > 0] = color
                
                vis_image = cv2.addWeighted(vis_image, 0.4, colored_mask, 0.6, 0)
        
        return vis_image
    
    def _get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """클래스 ID에 따른 고정 색상 반환 (BGR 형식) - 데이터셋별 최적화"""
        if self.dataset == 'cityscapes':
            return self._get_cityscapes_class_color(class_id)
        elif self.dataset == 'mapillary-vistas':
            return self._get_mapillary_vistas_class_color(class_id)
        else:
            return self._get_coco_class_color(class_id)
    
    def _get_cityscapes_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """Cityscapes 클래스별 고정 색상 (BGR 형식) - HuggingFace 공식 19개 클래스"""
        cityscapes_colors = {
            # 🛣️ 도로/교통 관련 - 매우 눈에 띄는 색상 (최우선!)
            0: (0, 255, 255),      # road - 밝은 노란색 (가장 중요!)
            1: (255, 0, 255),      # sidewalk - 핫 마젠타
            
            # 🏗️ 구조물/건물
            2: (70, 70, 70),       # building - 어두운 회색
            3: (102, 102, 156),    # wall - 회색-파랑
            4: (190, 153, 153),    # fence - 베이지
            5: (153, 153, 153),    # pole - 회색
            
            # 🚦 교통 시설
            6: (250, 170, 30),     # traffic light - 주황색
            7: (220, 220, 0),      # traffic sign - 노란색
            
            # 🌳 자연/환경
            8: (107, 142, 35),     # vegetation - 올리브 그린
            9: (152, 251, 152),    # terrain - 연한 초록
            10: (70, 130, 180),    # sky - 스카이 블루
            
            # 👥 사람/라이더
            11: (220, 20, 60),     # person - 크림슨
            12: (255, 0, 0),       # rider - 빨간색
            
            # 🚗 차량들
            13: (0, 0, 142),       # car - 파란색
            14: (0, 0, 70),        # truck - 어두운 파랑
            15: (0, 60, 100),      # bus - 네이비
            16: (0, 80, 100),      # train - 청록
            17: (0, 0, 230),       # motorcycle - 밝은 파랑
            18: (119, 11, 32),     # bicycle - 적갈색
        }
        
        # Cityscapes별 고정 색상 반환, 없으면 기본 색상 계산
        if class_id in cityscapes_colors:
            return cityscapes_colors[class_id]
        else:
            # 기본 색상 생성
            return self._generate_default_color(class_id)
    
    def _get_coco_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """COCO-Stuff 클래스별 고정 색상 (BGR 형식) - HuggingFace 공식 매핑 (0-170)"""
        class_colors = {
            # 🛣️ 도로/교통 관련 - 매우 눈에 띄는 색상 (최우선!)
            137: (0, 255, 255),    # road - 밝은 노란색 (가장 눈에 띔!)
            128: (255, 0, 255),    # pavement - 핫 마젠타 (매우 눈에 띔!)
            135: (0, 0, 255),      # railroad - 순수 빨간색
            133: (255, 128, 0),    # playingfield - 주황색 (운동장/도로 구분)
            
            # 🚗 교통수단 - 밝은 색상 (Things 클래스 0-79)
            2: (255, 20, 147),     # car - 딥핑크
            1: (0, 255, 127),      # bicycle - 스프링그린
            5: (255, 165, 0),      # bus - 주황색
            7: (255, 127, 0),      # truck - 오렌지
            3: (127, 255, 0),      # motorcycle - 라임
            
            # 👤 사람/동물
            0: (0, 255, 255),      # person - 노란색
            
            # 🌳 자연물 (HuggingFace 매핑 기준)
            157: (0, 255, 0),      # tree - 초록색
            112: (128, 255, 0),    # grass - 밝은 라임  
            145: (135, 206, 255),  # sky-other - 하늘색
            143: (255, 255, 0),    # sea - 시안
            136: (0, 255, 192),    # river - 아쿠아
            123: (139, 69, 19),    # mountain - 갈색
            99: (165, 42, 42),     # dirt - 브라운
            166: (64, 224, 208),   # water-other - 터콰이즈
            94: (135, 206, 250),   # clouds - 연한 하늘색
            107: (255, 182, 193),  # flower - 연핑크
            
            # 🏢 건물/구조물 (새로운 stuff ID 기준)
            96: (255, 0, 0),       # building-other - 파란색
            128: (255, 192, 203),  # house - 핑크
            173: (192, 192, 192),  # wall-other - 회색
            171: (169, 169, 169),  # wall-brick - 연회색
            175: (128, 128, 128),  # wall-stone - 어두운 회색
            176: (211, 211, 211),  # wall-tile - 밝은 회색
            177: (205, 133, 63),   # wall-wood - 나무색
            102: (255, 255, 255),  # ceiling-other - 흰색
            151: (139, 0, 139),    # roof - 다크 마젠타
            95: (70, 130, 180),    # bridge - 스틸 블루
            
            # 🪑 가구/실내
            86: (139, 0, 139),     # door-stuff - 다크 마젠타
            121: (173, 255, 47),   # table-merged - 그린옐로우
            56: (255, 105, 180),   # chair - 핫핑크
            120: (255, 165, 0),    # cabinet-merged - 주황색
            104: (127, 0, 255),    # shelf - 네온보라
            122: (218, 165, 32),   # floor-other-merged - 골든로드
            87: (160, 82, 45),     # floor-wood - 나무색
            
            # 🚦 교통 시설
            9: (255, 255, 0),      # traffic light - 노란색 (신호등)
            10: (255, 0, 0),       # fire hydrant - 빨간색 (소화전)
            11: (0, 0, 255),       # stop sign - 파란색 (정지 표지)
            
            # 🌸 기타 자연/장식
            88: (255, 182, 193),   # flower - 연핑크
            89: (255, 69, 0),      # fruit - 오렌지레드
            58: (34, 139, 34),     # potted plant - 포레스트그린
            
            # 🏠 기타 구조물  
            117: (128, 0, 128),    # fence-merged - 보라색
            92: (255, 215, 0),     # light - 골드
            114: (0, 191, 255),    # window-blind - 딥 스카이 블루
            115: (30, 144, 255),   # window-other - 닷저 블루
            85: (255, 215, 0),     # curtain - 골드
            
            # 📦 기타 객체들
            80: (220, 20, 60),     # banner - 크림슨
            81: (255, 240, 245),   # blanket - 라벤더 블러쉬
            83: (210, 180, 140),   # cardboard - 탄색
            84: (139, 69, 19),     # counter - 새들 브라운
            132: (255, 255, 127),  # rug-merged - 밝은 크림
        }
        
        # COCO 클래스별 고정 색상 반환, 없으면 기본 색상 계산
        if class_id in class_colors:
            return class_colors[class_id]
        else:
            return self._generate_default_color(class_id)
    
    def _get_mapillary_vistas_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """Mapillary Vistas 클래스별 고정 색상 (BGR 형식) - 공식 논문 66개 클래스"""
        mapillary_colors = {
            # 🐦 Animals
            0: (139, 69, 19),       # animal--bird - 갈색
            1: (160, 82, 45),       # animal--ground-animal - 새들브라운
            
            # 🚧 Construction - Barriers  
            2: (128, 64, 128),      # construction--barrier--curb - 보라색
            3: (190, 153, 153),     # construction--barrier--fence - 베이지
            4: (180, 165, 180),     # construction--barrier--guard-rail - 연보라
            5: (169, 169, 169),     # construction--barrier--other-barrier - 회색
            6: (102, 102, 156),     # construction--barrier--wall - 회색-파랑
            
            # 🛣️ Construction - Flat surfaces (가장 중요한 도로 관련!)
            7: (255, 20, 147),      # construction--flat--bike-lane - 핫핑크
            8: (255, 0, 255),       # construction--flat--crosswalk-plain - 마젠타
            9: (128, 0, 128),       # construction--flat--curb-cut - 자주색
            10: (255, 165, 0),      # construction--flat--driveway - 주황색
            11: (128, 64, 128),     # construction--flat--parking - 보라색
            12: (255, 128, 255),    # construction--flat--pedestrian-area - 연한 핑크
            13: (232, 35, 244),     # construction--flat--rail-track - 자주색
            14: (0, 255, 255),      # construction--flat--road - 밝은 노란색 (가장 중요!)
            15: (0, 255, 127),      # construction--flat--service-lane - 스프링그린
            16: (255, 0, 255),      # construction--flat--sidewalk - 핫 마젠타
            17: (255, 215, 0),      # construction--flat--traffic-island - 골드
            
            # 🏗️ Construction - Structures
            18: (150, 100, 100),    # construction--structure--bridge - 갈색
            19: (70, 70, 70),       # construction--structure--building - 어두운 회색
            20: (150, 120, 90),     # construction--structure--tunnel - 어두운 갈색
            
            # 👥 Humans
            21: (220, 20, 60),      # human--person - 크림슨
            22: (255, 0, 0),        # human--rider - 빨간색
            
            # ✏️ Markings
            23: (255, 255, 255),    # marking--crosswalk-zebra - 흰색
            24: (255, 255, 0),      # marking--general - 시안
            
            # 🌳 Nature
            25: (139, 69, 19),      # nature--mountain - 갈색
            26: (165, 42, 42),      # nature--sand - 브라운
            27: (70, 130, 180),     # nature--sky - 스카이 블루
            28: (255, 255, 255),    # nature--snow - 흰색
            29: (152, 251, 152),    # nature--terrain - 연한 초록
            30: (107, 142, 35),     # nature--vegetation - 올리브 그린
            31: (64, 224, 208),     # nature--water - 터쿼이즈
            
            # 🏪 Objects
            32: (220, 20, 60),      # object--banner - 크림슨
            33: (139, 0, 139),      # object--bench - 다크 마젠타
            34: (255, 69, 0),       # object--bike-rack - 오렌지레드
            35: (0, 0, 0),          # object--catch-basin - 검정
            36: (128, 128, 128),    # object--cctv-camera - 회색
            37: (255, 0, 0),        # object--fire-hydrant - 빨간색
            38: (128, 128, 128),    # object--junction-box - 회색
            39: (0, 0, 255),        # object--mailbox - 파란색
            40: (128, 128, 128),    # object--manhole - 회색
            41: (139, 0, 139),      # object--phone-booth - 다크 마젠타
            42: (64, 64, 64),       # object--pothole - 어두운 회색
            43: (255, 215, 0),      # object--street-light - 골드
            
            # 🏗️ Object - Support structures
            44: (153, 153, 153),    # object--support--pole - 회색
            45: (153, 153, 153),    # object--support--pole-group - 회색
            46: (220, 220, 0),      # object--support--traffic-sign-frame - 노란색
            47: (153, 153, 153),    # object--support--utility-pole - 회색
            
            # 🚦 Object - Traffic elements
            48: (250, 170, 30),     # object--traffic-light - 주황색
            49: (107, 142, 35),     # object--traffic-sign--back - 올리브 그린
            50: (220, 220, 0),      # object--traffic-sign--front - 노란색
            51: (139, 0, 139),      # object--trash-can - 다크 마젠타
            
            # 🚗 Object - Vehicles
            52: (119, 11, 32),      # object--vehicle--bicycle - 적갈색
            53: (0, 191, 255),      # object--vehicle--boat - 딥 스카이 블루
            54: (0, 60, 100),       # object--vehicle--bus - 네이비
            55: (0, 0, 142),        # object--vehicle--car - 파란색
            56: (0, 0, 90),         # object--vehicle--caravan - 어두운 파랑
            57: (0, 0, 230),        # object--vehicle--motorcycle - 밝은 파랑
            58: (0, 80, 100),       # object--vehicle--on-rails - 청록
            59: (255, 127, 80),     # object--vehicle--other-vehicle - 코랄
            60: (0, 0, 110),        # object--vehicle--trailer - 짙은 파랑
            61: (0, 0, 70),         # object--vehicle--truck - 어두운 파랑
            62: (255, 165, 0),      # object--vehicle--wheeled-slow - 주황색
            
            # ⚫ Void classes
            63: (81, 0, 81),        # void--ground - 자주색
            64: (64, 64, 64),       # void--static - 어두운 회색
            65: (0, 0, 0),          # void--unlabeled - 검정
        }
        
        # Mapillary Vistas별 고정 색상 반환, 없으면 기본 색상 계산
        if class_id in mapillary_colors:
            return mapillary_colors[class_id]
        else:
            # 기본 색상 생성
            return self._generate_default_color(class_id)
    
    def _generate_default_color(self, class_id: int) -> Tuple[int, int, int]:
        """기본 색상 생성 (HSV 기반)"""
        hue = (class_id * 137) % 180  # 137은 황금비율 근사치로 색상 분산
        color = np.array([[[hue, 255, 255]]], dtype=np.uint8)  # 최대 채도, 최대 명도
        color = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0][0]
        return (int(color[0]), int(color[1]), int(color[2]))
    
    def get_segmentation_stats(self, results: Dict) -> Dict:
        if not results:
            return {'total_objects': 0, 'classes': {}}
        
        labels = results.get('labels', [])
        
        if not labels:
            return {'total_objects': 0, 'classes': {}}
        
        class_counts = {}
        for label in labels:
            label_id = label.item() if torch.is_tensor(label) else label
            class_name = self.classes.get(label_id, f'class_{label_id}')
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        return {
            'total_objects': len(labels),
            'classes': class_counts
        }
    
    def _convert_semantic_to_instance_format(self, semantic_results):
        """Semantic segmentation 결과를 instance 형식으로 변환"""
        import torch
        import numpy as np
        
        segmentation = semantic_results
        
        # segmentation이 이미 numpy array인지 확인
        if torch.is_tensor(segmentation):
            segmentation = segmentation.cpu().numpy()
        
        # 각 클래스별로 마스크와 라벨 생성
        unique_labels = np.unique(segmentation)
        # 배경(0) 제거
        unique_labels = unique_labels[unique_labels > 0]
        
        masks = []
        labels = []
        scores = []
        
        for label_id in unique_labels:
            # 해당 클래스의 마스크 생성
            class_mask = (segmentation == label_id).astype(np.uint8)
            
            # 마스크가 충분히 큰 영역인지 확인 (너무 작은 영역 제외)
            if np.sum(class_mask) > 100:  # 100 픽셀 이상
                masks.append(torch.from_numpy(class_mask))
                labels.append(torch.tensor(int(label_id)))
                scores.append(torch.tensor(1.0))  # semantic은 confidence가 없으므로 1.0으로 설정
        
        return {
            'masks': masks,
            'labels': labels,
            'scores': scores,
            'segmentation': segmentation  # 원본 segmentation map 유지
        }