import cv2
import torch
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import os

model = YOLO('best.pt')

class PlayerTracker:
    def __init__(self):
        self.next_id = 0
        self.players = {}  
        self.max_frames_missing = 30  
        self.feature_extractor = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher()
        
    def _get_features(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        kp, des = self.feature_extractor.detectAndCompute(gray, None)
        return kp, des
        
    def _bbox_similarity(self, bbox1, bbox2):
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
    
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
    
        iou = intersection_area / union_area if union_area > 0 else 0
    
        aspect_ratio1 = (x2_1 - x1_1) / (y2_1 - y1_1) if (y2_1 - y1_1) > 0 else 0
        aspect_ratio2 = (x2_2 - x1_2) / (y2_2 - y1_2) if (y2_2 - y1_2) > 0 else 0
        aspect_sim = 1 - abs(aspect_ratio1 - aspect_ratio2) / max(aspect_ratio1, aspect_ratio2, 1e-6)
    
        area_sim = 1 - abs(bbox1_area - bbox2_area) / max(bbox1_area, bbox2_area, 1e-6)
    
        similarity_score = 0.6 * iou + 0.2 * aspect_sim + 0.2 * area_sim
        return similarity_score
        
    def update(self, detections, frame, frame_num):
        current_players = {}
        
        for det in detections:
            bbox = det[:4] 
            conf = det[4]
            
            best_match_id = None
            best_match_score = 0
            
            for pid, player in self.players.items():
                if frame_num - player['last_seen'] > self.max_frames_missing:
                    continue
                    
                bbox_sim = self._bbox_similarity(bbox, player['bbox'])
                feature_sim = 0
                
                if 'features' in player:
                    kp, des = self._get_features(frame, bbox)
                    if des is not None and player['features'] is not None:
                        matches = self.matcher.knnMatch(des, player['features'], k=2)
                        good = []
                        for m,n in matches:
                            if m.distance < 0.75*n.distance:
                                good.append([m])
                        feature_sim = len(good) / min(len(des), len(player['features']))
                
                total_score = 0.7 * bbox_sim + 0.3 * feature_sim
                if total_score > best_match_score and total_score > 0.5:
                    best_match_score = total_score
                    best_match_id = pid
            
            if best_match_id is not None:
                pid = best_match_id
                self.players[pid]['bbox'] = bbox
                self.players[pid]['last_seen'] = frame_num
                kp, des = self._get_features(frame, bbox)
                if des is not None:
                    self.players[pid]['features'] = des
            else:
                pid = self.next_id
                self.next_id += 1
                kp, des = self._get_features(frame, bbox)
                self.players[pid] = {
                    'bbox': bbox,
                    'features': des,
                    'last_seen': frame_num
                }
            
            current_players[pid] = bbox
        
        to_delete = [pid for pid, p in self.players.items() 
                    if frame_num - p['last_seen'] > self.max_frames_missing]
        for pid in to_delete:
            del self.players[pid]
            
        return current_players
def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    tracker = PlayerTracker()
    frame_num = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame)
        detections = []
        for result in results:
            for box in result.boxes:
                if result.names[int(box.cls)] == 'player':
                    bbox = box.xyxy[0].cpu().numpy()
                    conf = box.conf.item()  
                    detections.append(np.array([*bbox, conf]))
        
        current_players = tracker.update(detections, frame, frame_num)
        
        for pid, bbox in current_players.items():
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Player {pid}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        out.write(frame)
        frame_num += 1
        
        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

print("File exists:", os.path.exists('15sec_input_720p.mp4'))
process_video('15sec_input_720p.mp4', 'output_tracked.mp4')
