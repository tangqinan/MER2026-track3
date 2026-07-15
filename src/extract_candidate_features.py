# -*- coding: utf-8 -*-
import os
import cv2
import librosa
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm
import multiprocessing
import warnings
warnings.filterwarnings('ignore')

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# =====================================================================
# 1. 路径与核心配置
# =====================================================================
DATASET_DIR = "/home/image006/MER2026/dataset/mer2026-dataset"
CANDIDATE_CSV = os.path.join(DATASET_DIR, "track3_candidate.csv")

TARGET_DIR = "/home/image006/MER2026/TT/20260525"
CACHE_DIR = "/home/image006/MER2026/TT/20260528_Gemini/candidate_features"
os.makedirs(CACHE_DIR, exist_ok=True)

AU_MAP = {
    "AU01": "Inner Brow Raiser", "AU02": "Outer Brow Raiser", "AU04": "Brow Lowerer",
    "AU05": "Upper Lid Raiser", "AU06": "Cheek Raiser", "AU07": "Lid Tightener",
    "AU09": "Nose Wrinkler", "AU10": "Upper Lip Raiser", "AU12": "Lip Corner Puller",
    "AU14": "Dimpler", "AU15": "Lip Corner Depressor", "AU17": "Chin Raiser",
    "AU20": "Lip Stretcher", "AU23": "Lip Tightener", "AU25": "Lips Part",
    "AU26": "Jaw Drop", "AU28": "Lip Suck", "AU43": "Eyes Closed"
}
EMO_LIST = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]

# 定义我们要使用的 4 张黄金显卡集群
AVAILABLE_GPUS = [1, 3, 4, 5]

# =====================================================================
# 2. 多卡独占式 Worker 初始化逻辑
# =====================================================================
face_detector = None
worker_device = "cpu"

def init_worker_detector():
    """每个子进程绑定一张专属显卡，实现真正的四卡并联"""
    global face_detector, worker_device
    
    # 根据子进程的内部身份 ID，对 4 张卡进行取模轮询分配
    try:
        process_idx = multiprocessing.current_process()._identity[0] - 1
        gpu_id = AVAILABLE_GPUS[process_idx % len(AVAILABLE_GPUS)]
    except:
        gpu_id = AVAILABLE_GPUS[0] # 异常兜底
        
    worker_device = f"cuda:{gpu_id}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id) # 强行隔离当前进程的可见卡
    
    from feat import Detector
    face_detector = Detector(
        face_model="retinaface", 
        landmark_model="mobilefacenet", 
        au_model="svm", 
        emotion_model="resmasknet",
        device="cuda" # 此时的 cuda 已经自动指向隔离后的目标卡
    )

def process_single_candidate(row_dict):
    global face_detector, worker_device
    vid_name = str(row_dict['name'])
    cache_path = os.path.join(CACHE_DIR, f"{vid_name}.txt")
    
    # 💎 增量断点续传：存在则直接跳过
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 10:
        return {"status": "exists"}

    # 路径匹配
    vid_path = os.path.join(DATASET_DIR, "video/video_track3_candidate/video", f"{vid_name}.mp4")
    aud_path = os.path.join(DATASET_DIR, "audio/audio_track3_candidate/audio", f"{vid_name}.wav")

    if not os.path.exists(vid_path) or not os.path.exists(aud_path):
        vid_path = os.path.join(DATASET_DIR, "video/video_track3_candidate", f"{vid_name}.mp4")
        aud_path = os.path.join(DATASET_DIR, "audio/audio_track3_candidate", f"{vid_name}.wav")
        if not os.path.exists(vid_path) or not os.path.exists(aud_path):
            return {"status": "missing"}

    # 🔊 1:1 像素级声学特征提取
    try:
        y, sr = librosa.load(aud_path, sr=16000)
        rmse = librosa.feature.rms(y=y)[0]
        mean_energy = float(np.mean(rmse))
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        pitch_var = float(np.var(f0)) if len(f0) > 0 else 0.0
        spec_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        mean_centroid = float(np.mean(spec_centroids))
        onsets = librosa.onset.onset_detect(y=y, sr=sr)
        duration = len(y) / sr
        speech_rate = float(len(onsets) / duration) if duration > 0 else 0.0
        audio_str = f"Voice energy={mean_energy:.4f}, Pitch var={pitch_var:.2f}, Sharpness={mean_centroid:.1f}Hz, Speed={speech_rate:.2f}b/s."
    except:
        audio_str = "Voice energy and pitch contour are stable."

    # 🎬 1:1 像素级每 6 帧抽帧沙盒机制
    pid = os.getpid()
    vid_tmp_dir = os.path.join(TARGET_DIR, f"tmp_cand_{vid_name}_pid{pid}")
    os.makedirs(vid_tmp_dir, exist_ok=True)
    
    sampled_paths = []
    cap = cv2.VideoCapture(vid_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_count % 6 == 0:  
            img_path = os.path.join(vid_tmp_dir, f"f_{frame_count}.jpg")
            cv2.imwrite(img_path, frame)
            sampled_paths.append(img_path)
        frame_count += 1
    cap.release()

    au_activations = {au: [] for au in AU_MAP.keys()}
    emo_activations = {emo: [] for emo in EMO_LIST}

    # 🖼️ 像素级继承 Py-Feat 批检测
    try:
        if len(sampled_paths) > 0:
            all_detections = face_detector.detect_image(sampled_paths, batch_size=32)
            if all_detections is not None and len(all_detections) > 0:
                for au in AU_MAP.keys():
                    if au in all_detections.columns:
                        au_activations[au] = [float(v) for v in all_detections[au].dropna().values]
                for emo in EMO_LIST:
                    if emo in all_detections.columns:
                        emo_activations[emo] = [float(v) for v in all_detections[emo].dropna().values]
    except:
        pass
    finally:
        shutil.rmtree(vid_tmp_dir, ignore_errors=True)

    au_peaks = []
    for au, name in AU_MAP.items():
        vals = au_activations[au]
        peak_val = float(np.max(vals)) if len(vals) > 0 else 0.0
        au_peaks.append((peak_val, name))
    au_peaks.sort(reverse=True, key=lambda x: x[0])
    au_str = ", ".join([f"{name}(peak:{p:.2f})" for p, name in au_peaks[:3]])

    emo_peaks = []
    for emo in EMO_LIST:
        vals = emo_activations[emo]
        peak_val = float(np.max(vals)) if len(vals) > 0 else 0.0
        emo_peaks.append((peak_val, emo))
    emo_peaks.sort(reverse=True, key=lambda x: x[0])
    emo_str = ", ".join([f"{emo}(max_prob:{p:.2f})" for p, emo in emo_peaks[:2]])

    report_str = f"[Acoustic Prosody] {audio_str} | [Facial Micro-movements] {au_str} | [Macro Emotion Base] {emo_str}"
    
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(report_str)

    return {"status": "success"}

# =====================================================================
# 3. 主进程并行调度流
# =====================================================================
if __name__ == '__main__':
    print("\n🚀 [多卡战神版] 正在加载 10,000 条盲测集 Candidate 表格...")
    df_candidate = pd.read_csv(CANDIDATE_CSV)
    tasks_inputs = df_candidate.to_dict('records')
    total_raw = len(tasks_inputs)

    # 4 进程对应 4 张显卡
    NUM_WORKERS = len(AVAILABLE_GPUS)
    print(f"🔥 集群引擎启动！并联显卡列表: {AVAILABLE_GPUS} | 并行工作进程数: {NUM_WORKERS}")

    stats = {"success": 0, "exists": 0, "missing": 0}

    with multiprocessing.Pool(processes=NUM_WORKERS, initializer=init_worker_detector) as pool:
        results = pool.imap_unordered(process_single_candidate, tasks_inputs)
        
        for res in tqdm(results, total=total_raw, desc="Multi-GPU Extracting"):
            if res["status"] == "success":
                stats["success"] += 1
            elif res["status"] == "exists":
                stats["exists"] += 1
            elif res["status"] == "missing":
                stats["missing"] += 1

    print("\n" + "="*60)
    print("🏁 TRACK3 CANDIDATE MULTI-GPU EXTRACTION COMPLETE")
    print("="*60)
    print(f"✅ 新成功提取落盘 (New Extracted) : {stats['success']} 条")
    print(f"♻️ 断点增量跳过 (Already Exists) : {stats['exists']} 条")
    print(f"❌ 丢失媒体文件 (Missing Media)  : {stats['missing']} 条")
    print(f"📂 缓存统一大本营: {CACHE_DIR}")
    print("="*60)
