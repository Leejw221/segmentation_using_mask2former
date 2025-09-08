#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
import argparse
import time
import logging
from pathlib import Path
import glob

# 현재 파일 위치를 기준으로 모듈 경로 설정
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

from hf_model_manager import HuggingFaceModelManager
from hf_segmentation_processor import HuggingFaceSegmentationProcessor

class SegM2F:
    def __init__(self, config_path=None):
        """Mask2Former Segmentation 클래스"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 설정 파일 경로 설정
        if config_path is None:
            config_path = current_dir / 'config' / 'hf_model_config.yaml'
        
        self.logger.info(f"Loading config from: {config_path}")
        
        # 모델 매니저 및 프로세서 초기화
        self.model_manager = HuggingFaceModelManager(str(config_path))
        self.segmentation_processor = HuggingFaceSegmentationProcessor(self.model_manager)
        
        # 모델 로드
        if not self.model_manager.load_model():
            raise RuntimeError("Failed to load segmentation model")
        
        self.logger.info("✅ Seg-M2F initialized successfully!")
        
    def process_image(self, image_path):
        """단일 이미지 처리"""
        image = cv2.imread(str(image_path))
        if image is None:
            self.logger.error(f"Could not load image: {image_path}")
            return None, None
            
        self.logger.info(f"Processing: {image_path}")
        
        # 세그멘테이션 수행
        results, segmentation_vis = self.segmentation_processor.process_image(image)
        
        # 통계 출력 (간소화)
        stats = self.segmentation_processor.get_segmentation_stats(results)
        self.logger.info(f"Detected {stats['total_objects']} objects")
        
        return image, segmentation_vis, results
    
    def process_image_folder(self, input_dir, output_dir, mode='both'):
        """이미지 폴더 전체 처리 - 시간 정보 포함하여 저장"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 이미지 파일 찾기
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(ext))
            image_files.extend(input_path.glob(ext.upper()))
        
        self.logger.info(f"Found {len(image_files)} images in {input_path}")
        
        for i, image_file in enumerate(image_files):
            self.logger.info(f"[{i+1}/{len(image_files)}] Processing: {image_file.name}")
            
            original, segmentation, results = self.process_image(image_file)
            if original is None:
                continue
            
            # 현재 시간 정보 및 데이터셋 정보
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = self.model_manager.config['model'].get('dataset', 'coco')
            base_name = f"{timestamp}_{dataset_name}_{image_file.stem}"
            
            # 세그멘테이션 결과만 저장 (원본 + 세그멘테이션 합성)
            segmentation_overlay = self._create_overlay(original, segmentation, alpha=0.8)
            seg_file = output_path / f"{base_name}_segmentation{image_file.suffix}"
            cv2.imwrite(str(seg_file), segmentation_overlay)
            
            # 인식된 클래스 정보를 텍스트 파일로 저장
            class_info_file = output_path / f"{base_name}_classes.txt"
            self._save_class_info_to_file(class_info_file, image_file.name, results)
            
            self.logger.info(f"✅ Saved: {base_name}_segmentation & {base_name}_classes.txt")
    
    def _create_overlay(self, original, segmentation, alpha=1.00):
        """원본 이미지와 세그멘테이션 오버레이"""
        return cv2.addWeighted(original, 1-alpha, segmentation, alpha, 0)
    
    def process_video(self, input_video, output_dir):
        """비디오 파일 처리 - 원본과 세그멘테이션 영상 별도 저장"""
        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            self.logger.error(f"Could not open video: {input_video}")
            return
        
        # 비디오 정보
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 출력 디렉토리 생성
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 현재 시간 정보 및 데이터셋 정보
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = self.model_manager.config['model'].get('dataset', 'coco')
        input_name = Path(input_video).stem
        
        # 비디오 라이터 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # 세그멘테이션 비디오 저장용 (원본 + 세그멘테이션 합성)
        seg_output = output_path / f"{timestamp}_{dataset_name}_{input_name}_segmentation.mp4"
        out_segmentation = cv2.VideoWriter(str(seg_output), fourcc, fps, (width, height))
        
        self.logger.info(f"Processing video: {total_frames} frames at {fps:.1f} FPS")
        
        frame_count = 0
        start_time = time.time()
        video_class_info = []  # 영상의 모든 프레임 클래스 정보 저장
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 세그멘테이션 수행
            results, segmentation_vis = self.segmentation_processor.process_image(frame)
            
            # 프레임별 클래스 정보 저장 (시간 포함)
            current_time = frame_count / fps
            frame_class_info = self._extract_frame_class_info(current_time, results)
            if frame_class_info['classes']:  # 클래스가 감지된 경우만 저장
                video_class_info.append(frame_class_info)
            
            # 세그멘테이션 프레임 저장 (원본 + 세그멘테이션 합성)
            segmentation_overlay = self._create_overlay(frame, segmentation_vis, alpha=0.9)
            out_segmentation.write(segmentation_overlay)
            
            # 진행상황 출력
            if frame_count % 30 == 0:  # 30프레임마다
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed
                eta = (total_frames - frame_count) / fps_current if fps_current > 0 else 0
                self.logger.info(f"Progress: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%) "
                               f"FPS: {fps_current:.1f}, ETA: {eta:.1f}s")
        
        cap.release()
        out_segmentation.release()
        
        # 영상 클래스 정보를 텍스트 파일로 저장
        class_info_file = output_path / f"{timestamp}_{dataset_name}_{input_name}_classes.txt"
        self._save_video_class_info_to_file(class_info_file, input_name, video_class_info, fps, total_frames)
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        self.logger.info(f"✅ Video processing complete!")
        self.logger.info(f"   Segmentation: {seg_output}")
        self.logger.info(f"   Classes Info: {class_info_file}")
        self.logger.info(f"   Average FPS: {avg_fps:.1f}")

    def _save_class_info_to_file(self, file_path, image_name, results):
        """이미지의 인식된 클래스 정보를 텍스트 파일로 저장"""
        try:
            stats = self.segmentation_processor.get_segmentation_stats(results)
            dataset_name = self.model_manager.config['model'].get('dataset', 'coco')
            model_name = self.model_manager.model_name
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write(f"SEGMENTATION RESULTS - IMAGE\n")
                f.write("=" * 60 + "\n")
                f.write(f"Image: {image_name}\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Total Objects Detected: {stats['total_objects']}\n")
                f.write(f"Processing Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n" + "-" * 60 + "\n")
                f.write("DETECTED CLASSES:\n")
                f.write("-" * 60 + "\n")
                
                if stats['classes']:
                    for class_name, count in sorted(stats['classes'].items()):
                        f.write(f"{class_name}: {count} object(s)\n")
                else:
                    f.write("No objects detected.\n")
                
                f.write("\n" + "=" * 60 + "\n")
                
        except Exception as e:
            self.logger.warning(f"Failed to save class info: {e}")

    def _extract_frame_class_info(self, timestamp, results):
        """프레임의 클래스 정보 추출"""
        stats = self.segmentation_processor.get_segmentation_stats(results)
        return {
            'timestamp': timestamp,
            'total_objects': stats['total_objects'],
            'classes': stats['classes']
        }

    def _save_video_class_info_to_file(self, file_path, video_name, video_class_info, fps, total_frames):
        """영상의 인식된 클래스 정보를 텍스트 파일로 저장"""
        try:
            dataset_name = self.model_manager.config['model'].get('dataset', 'coco')
            model_name = self.model_manager.model_name
            
            # 전체 클래스 통계 계산
            all_classes = {}
            total_objects_detected = 0
            frames_with_detection = len(video_class_info)
            
            for frame_info in video_class_info:
                total_objects_detected += frame_info['total_objects']
                for class_name, count in frame_info['classes'].items():
                    all_classes[class_name] = all_classes.get(class_name, 0) + count
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"SEGMENTATION RESULTS - VIDEO\n")
                f.write("=" * 80 + "\n")
                f.write(f"Video: {video_name}\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Model: {model_name}\n")
                f.write(f"Total Frames: {total_frames}\n")
                f.write(f"Video FPS: {fps:.1f}\n")
                f.write(f"Video Duration: {total_frames/fps:.1f} seconds\n")
                f.write(f"Frames with Detection: {frames_with_detection}\n")
                f.write(f"Total Objects Detected: {total_objects_detected}\n")
                f.write(f"Processing Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                f.write("\n" + "-" * 80 + "\n")
                f.write("OVERALL CLASS STATISTICS:\n")
                f.write("-" * 80 + "\n")
                
                if all_classes:
                    for class_name, total_count in sorted(all_classes.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"{class_name}: {total_count} detections\n")
                else:
                    f.write("No objects detected in any frame.\n")
                
                f.write("\n" + "-" * 80 + "\n")
                f.write("FRAME-BY-FRAME DETECTION LOG:\n")
                f.write("-" * 80 + "\n")
                
                if video_class_info:
                    for frame_info in video_class_info:
                        timestamp = frame_info['timestamp']
                        minutes = int(timestamp // 60)
                        seconds = timestamp % 60
                        
                        f.write(f"Time: {minutes:02d}:{seconds:06.3f} | Objects: {frame_info['total_objects']} | ")
                        
                        class_list = []
                        for class_name, count in frame_info['classes'].items():
                            class_list.append(f"{class_name}({count})")
                        
                        f.write("Classes: " + ", ".join(class_list) + "\n")
                else:
                    f.write("No detections found in any frame.\n")
                
                f.write("\n" + "=" * 80 + "\n")
                
        except Exception as e:
            self.logger.warning(f"Failed to save video class info: {e}")


def main():
    parser = argparse.ArgumentParser(description='Mask2Former Segmentation Tool')
    parser.add_argument('mode', choices=['image', 'video'], 
                       help='Processing mode: image or video')
    parser.add_argument('-c', '--config', 
                       help='Config file path (optional)')
    parser.add_argument('-m', '--model', 
                       help='Model name to use (overrides config)')
    parser.add_argument('-d', '--dataset', choices=['coco', 'ade20k', 'cityscapes'],
                       help='Dataset type: coco (general objects), ade20k (scenes), cityscapes (driving)')
    
    args = parser.parse_args()
    
    # 기본 경로 설정
    current_dir = Path(__file__).parent.absolute()
    dataset_dir = current_dir / 'dataset'
    result_dir = current_dir / 'result'
    
    try:
        # Seg-M2F 초기화
        seg_m2f = SegM2F(args.config)
        
        # 모델 변경 (옵션)
        if args.model:
            print(f"🔄 Switching to model: {args.model}")
            if not seg_m2f.model_manager.load_model(args.model):
                print(f"❌ Failed to load model: {args.model}")
                sys.exit(1)
            print("✅ Model switched successfully!")
        
        # 데이터셋 변경 (옵션)
        if args.dataset:
            print(f"📊 Setting dataset to: {args.dataset}")
            seg_m2f.model_manager.config['model']['dataset'] = args.dataset
            # 프로세서 재초기화
            seg_m2f.segmentation_processor = HuggingFaceSegmentationProcessor(seg_m2f.model_manager)
            print("✅ Dataset switched successfully!")
        
        # 현재 설정 출력
        dataset = seg_m2f.model_manager.config['model'].get('dataset', 'coco')
        print(f"🎯 Current model: {seg_m2f.model_manager.model_name}")
        print(f"📊 Current dataset: {dataset}")
        print(f"🖥️  Device: {seg_m2f.model_manager.device}")
        
        if args.mode == 'image':
            # 이미지 폴더 전체 처리
            image_input_dir = dataset_dir / 'image'
            image_output_dir = result_dir / 'image' 
            
            if not image_input_dir.exists():
                print(f"❌ Image input directory not found: {image_input_dir}")
                print(f"   Please create directory and add images: mkdir -p {image_input_dir}")
                sys.exit(1)
            
            print(f"📂 Processing images from: {image_input_dir}")
            print(f"💾 Saving results to: {image_output_dir}")
            seg_m2f.process_image_folder(str(image_input_dir), str(image_output_dir))
        
        elif args.mode == 'video':
            # 비디오 폴더 처리 (dataset/video에서 입력)
            video_input_dir = dataset_dir / 'video'
            video_output_dir = result_dir / 'video'
            
            if not video_input_dir.exists():
                print(f"❌ Video input directory not found: {video_input_dir}")
                print(f"   Please create directory and add MP4 files:")
                print(f"   1. Convert bags: cd ../bag2mp4 && python3 bag_to_mp4.py")
                print(f"   2. Copy MP4s: cp ../bag2mp4/video/*.mp4 {video_input_dir}/")
                print(f"   3. Or add MP4 files directly to: {video_input_dir}/")
                sys.exit(1)
            
            input_dir = video_input_dir
            
            # 비디오/bag 파일 찾기 (재귀적으로 하위 폴더도 검색)
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.bag', '*.db3']
            video_files = []
            
            # 일반 파일과 하위 디렉토리 모두 검색
            for ext in video_extensions:
                video_files.extend(input_dir.glob(ext))
                video_files.extend(input_dir.glob(ext.upper()))
                video_files.extend(input_dir.rglob(ext))  # 재귀적 검색
                video_files.extend(input_dir.rglob(ext.upper()))
            
            # 중복 제거
            video_files = list(set(video_files))
            
            if not video_files:
                print(f"❌ No video/bag files found in: {input_dir}")
                print(f"   Searched recursively for: mp4, avi, mov, mkv, bag")
                print(f"   Directory contents:")
                for item in input_dir.rglob('*'):
                    if item.is_file():
                        print(f"     {item}")
                sys.exit(1)
            
            print(f"🎬 Processing {len(video_files)} file(s) from: {input_dir}")
            print(f"💾 Saving results to: {video_output_dir}")
            for vf in video_files:
                print(f"   📁 Found: {vf}")
            
            for video_file in video_files:
                print(f"\n🔄 Processing: {video_file.name}")
                seg_m2f.process_video(str(video_file), str(video_output_dir))
            
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()