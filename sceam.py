# scream_panns.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import math

from pathlib import Path
import random

import sounddevice as sd
import numpy as np

from datetime import datetime
import socket
import json

# 🔹 PANNs Cnn14 가져오기 (panns_transfer 레포 기준)
#   - 레포 구조에 따라 import 경로는 수정 필요할 수 있습니다.
from panns_transfer_to_gtzan.pytorch.models import Cnn14   # 예: panns_transfer/models.py 안에 있음


# =========================
#  Dataset (waveform 출력)
# =========================
class ScreamWaveformDataset(Dataset):
    """
    root_dir 안에
      - scream/ *.wav   (label=1)
      - non_scream/ *.wav (label=0)
    구조로 되어 있다고 가정합니다.
    PANNs 쪽에서 log-mel을 계산하므로 여기서는 waveform만 반환합니다.

    ✅ 변경점:
    - 한 파일이 2초보다 길면, 2초 단위로 여러 chunk로 나눠서
      데이터셋 샘플을 늘림.
      예) 5초짜리 → 2초씩 3개(chunk)로 사용 (마지막은 padding)
    """

    def __init__(self, root_dir,
                 sample_rate=32000,  # PANNs 기본 32kHz
                 duration=2.0,
                 is_train=True):
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        self.is_train = is_train

        # (path, chunk_idx, label) 형태로 저장
        self.chunks = []

        for label_name, label in [('non_scream', 0), ('scream', 1)]:
            class_dir = self.root_dir / label_name
            if not class_dir.exists():
                continue

            for wav_path in class_dir.rglob('*.wav'):
                # 파일 길이 확인 위해 한 번 로드 (dataset 생성 시 1회)
                wav, sr = torchaudio.load(str(wav_path))

                # mono
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)

                # resample to target sr
                if sr != self.sample_rate:
                    wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

                total_len = wav.shape[1]
                if total_len <= 0:
                    continue  # 빈 파일 방지

                # 몇 개의 2초 chunk를 만들지 결정
                num_chunks = max(1, math.ceil(total_len / self.num_samples))

                for chunk_idx in range(num_chunks):
                    self.chunks.append((wav_path, chunk_idx, label))

        print(f"[{root_dir}] 총 파일 기반 chunk 수: {len(self.chunks)}")

    def __len__(self):
        return len(self.chunks)

    def _load_audio(self, path, chunk_idx):
        """
        지정된 chunk_idx에 해당하는 2초짜리 구간만 잘라서 반환
        """
        wav, sr = torchaudio.load(str(path))  # (C, T)

        # mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # resample to 32k
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        total_len = wav.shape[1]

        # 해당 chunk의 시작/끝 샘플 인덱스
        start = chunk_idx * self.num_samples
        end = start + self.num_samples

        # start가 파일 길이를 넘지 않도록 보호
        if start >= total_len:
            # 안전장치: 마지막 부분으로 강제 이동
            start = max(0, total_len - self.num_samples)
            end = start + self.num_samples

        # 마지막 chunk는 padding이 필요할 수 있음
        if end > total_len:
            pad_len = end - total_len
            wav = F.pad(wav, (0, pad_len))
            total_len = wav.shape[1]

        wav = wav[:, start:end]  # (1, num_samples)

        return wav

    def _augment(self, wav):
        # 간단한 증강: gain + 잡음
        if random.random() < 0.5:
            wav = wav * (0.5 + random.random())  # 0.5~1.5배
        if random.random() < 0.5:
            noise = torch.randn_like(wav) * 0.003
            wav = wav + noise
        return wav

    def __getitem__(self, idx):
        path, chunk_idx, label = self.chunks[idx]

        wav = self._load_audio(path, chunk_idx)

        if self.is_train:
            wav = self._augment(wav)

        # PANNs 구현에 따라 (B, T) 또는 (B, 1, T)를 받을 수 있습니다.
        # 여기서는 (T,) 형태를 반환하고, collate 후 (B, T)로 사용할 예정
        wav = wav.squeeze(0)  # (T,)

        return wav, torch.tensor(label, dtype=torch.float32)


# =========================
#  PANNs 기반 비명 탐지 모델
# =========================
class PANNsScreamModel(nn.Module):
    """
    PANNs Cnn14 backbone + binary classifier head
    """

    def __init__(self,
                 sample_rate=32000,
                 window_size=1024,
                 hop_size=320,
                 mel_bins=64,
                 fmin=50,
                 fmax=14000,
                 classes_num=527,
                 pretrained_checkpoint: str = None,
                 freeze_backbone: bool = True):
        super().__init__()

        # Cnn14 backbone
        self.backbone = Cnn14(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            classes_num=classes_num,
        )

        # 사전학습 weight 로드 (AudioSet)
        if pretrained_checkpoint is not None:
            ckpt = torch.load(pretrained_checkpoint, map_location='cpu')
            # 레포에서 제공하는 키 이름에 맞게 조정 필요
            state_dict = ckpt.get('model', ckpt)
            self.backbone.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights from: {pretrained_checkpoint}")

        # backbone freeze (원하면 풀어서 end-to-end fine-tune)
        if freeze_backbone:
            print('++++')
            for p in self.backbone.parameters():
                p.requires_grad = False

            print('----')

        # Cnn14의 임베딩 차원은 2048 (레포 기준)
        self.head = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)  # binary classification (scream / non_scream)
        )

    def forward(self, x):
        """
        x: waveform tensor, shape = (B, T)
           (필요시 (B, 1, T)에서 squeeze 해서 사용)
        """
        if x.dim() == 3:
            # (B, 1, T) -> (B, T)
            x = x.squeeze(1)

        # PANNs Cnn14의 forward는 (waveform, mixup_lambda=None) 형태인 경우가 많습니다.
        # 실제 구현에 따라 인자 형식은 조정 필요합니다.
        out_dict = self.backbone(x, None)
        embedding = out_dict['embedding']  # (B, 2048)

        logit = self.head(embedding).squeeze(1)  # (B,)
        return logit


# =========================
#  Train / Eval 루프
# =========================
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    bce = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_correct = 0
    total = 0

    for wav, y in loader:
        # wav: (B, T)
        wav = wav.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(wav)
        loss = bce(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * wav.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += (preds == y).sum().item()
        total += wav.size(0)

        print('train loss : ', (total_loss / total), end='\r')

    return total_loss / total, total_correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    bce = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_correct = 0
    total = 0

    for wav, y in loader:
        wav = wav.to(device)
        y = y.to(device)

        logits = model(wav)
        loss = bce(logits, y)

        total_loss += loss.item() * wav.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += (preds == y).sum().item()
        total += wav.size(0)

        print('valid loss : ', total_loss / total, end='\r')

    return total_loss / total, total_correct / total


# =========================
#  단일 파일 추론
# =========================
@torch.no_grad()
def detect_scream_panns(model,
                        wav_path,
                        device,
                        sample_rate=32000,
                        duration=2.0):
    model.eval()
    wav_path = Path(wav_path)

    num_samples = int(sample_rate * duration)

    wav, sr = torchaudio.load(str(wav_path))  # (C, T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)

    if wav.shape[1] < num_samples:
        pad = num_samples - wav.shape[1]
        left = pad // 2
        right = pad - left
        wav = F.pad(wav, (left, right))
    elif wav.shape[1] > num_samples:
        start = (wav.shape[1] - num_samples) // 2
        wav = wav[:, start:start + num_samples]

    wav = wav.squeeze(0).unsqueeze(0).to(device)  # (1, T)

    logit = model(wav)
    prob = torch.sigmoid(logit).item()
    is_scream = prob >= 0.5

    return is_scream, prob


# =========================
#  main: 학습 스크립트
# =========================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    print(1)
    train_ds = ScreamWaveformDataset('./data/train', is_train=True)
    print(2)
    val_ds   = ScreamWaveformDataset('./data/val',   is_train=False)
    print(3)

    train_loader = DataLoader(train_ds, batch_size=32,
                              shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=32,
                              shuffle=False, num_workers=4)

    # 🔹 PANNs 기반 모델 생성
    model = PANNsScreamModel(
        pretrained_checkpoint='./Cnn14.pth',  # 실제 경로/파일명으로 수정
        freeze_backbone=True
    ).to(device)


    # backbone을 freeze했다면 head만 학습되므로 학습 속도/안정성이 좋습니다.
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        model.parameters()),
                                 lr=1e-3)

    best_val_acc = 0.0
    num_epochs = 500

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, device)

        print(f'Epoch {epoch:02d} | '
              f'train_loss={train_loss:.4f} acc={train_acc:.3f} | '
              f'val_loss={val_loss:.4f} acc={val_acc:.3f}')

        # if val_acc > best_val_acc:
        if True:
            torch.save(model.state_dict(), f'check_point/panns_scream_best_{epoch}.pt')
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f'  >> Best model saved (val_acc={best_val_acc:.3f})')

    print("Training finished. Best val_acc:", best_val_acc)


@torch.no_grad()
def run_realtime_scream_detection_sliding(
    # checkpoint_path='panns_scream_best.pt',
    checkpoint_path='panns_scream_best - 복사본.pt',
    sample_rate=32000,
    window_duration=2.0,   # 모델이 보는 길이(2초)
    hop_duration=0.5,      # 판정 주기(0.5초)
    threshold=0.7,
    device_str=None,
):
    """
    2초짜리 분석 윈도우를 0.5초마다 굴리는 슬라이딩 윈도우 방식 실시간 비명 감지.

    - checkpoint_path : 학습 완료된 모델 가중치(panns_scream_best.pt)
    - sample_rate     : 32000 (학습 시와 동일)
    - window_duration : 한 번에 모델이 보는 길이 (2초)
    - hop_duration    : 판정 간격 (0.5초 → latency)
    - threshold       : 비명이라고 판단할 기준 확률
    """

    UDP_IP = "127.0.0.1"  # 받는 쪽 IP
    UDP_PORT = 2301  # 받는 쪽 포트

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # 디바이스 선택
    if device_str is None:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    print(f"[Realtime-Sliding] Using device: {device}")

    # 1) 모델 생성 및 가중치 로드
    model = PANNsScreamModel(
        sample_rate=sample_rate,
        pretrained_checkpoint=None,   # 체크포인트에 이미 backbone+head 저장된 상태
        freeze_backbone=True,
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2) 윈도우 / hop 샘플 수 계산
    window_samples = int(sample_rate * window_duration)   # 2초 → 64000 (32kHz 기준)
    hop_samples    = int(sample_rate * hop_duration)      # 0.5초 → 16000

    # print(f"[Realtime-Sliding] window={window_duration:.2f}s ({window_samples} samples), "
    #       f"hop={hop_duration:.2f}s ({hop_samples} samples)")
    # print("  마이크 입력을 0.5초마다 받아서, 항상 최근 2초를 모델에 넣어 감지합니다.")
    # print("  종료하려면 Ctrl+C 를 누르세요.\n")

    # 3) 최근 2초 버퍼 초기화 (처음에는 0으로 채움)
    buffer = torch.zeros(window_samples, device=device)  # (T,)
    t1 = None
    detection = False

    try:
        while True:
            # 4) 마이크에서 0.5초 분량 녹음
            print("🎧 Listening ...", end="\r")
            audio = sd.rec(
                frames=hop_samples,
                samplerate=sample_rate,
                channels=1,
                dtype='float32',
            )
            sd.wait()  # 0.5초 대기

            # audio: (hop_samples, 1) -> (hop_samples,)
            new_block = audio.squeeze(1)     # numpy (hop_samples,)
            new_block = torch.from_numpy(new_block).to(device)  # (hop_samples,)

            # 만약 녹음 길이가 부족하면 패딩
            if new_block.numel() < hop_samples:
                pad_len = hop_samples - new_block.numel()
                new_block = F.pad(new_block, (0, pad_len))

            # 5) 버퍼를 왼쪽으로 hop만큼 밀고, 뒤에 새 블록 붙이기
            buffer = torch.cat([buffer[hop_samples:], new_block], dim=0)  # 여전히 (window_samples,)



            # 6) 현재 버퍼(최근 2초)를 모델에 입력
            wav = buffer.unsqueeze(0)  # (1, T)
            logit = model(wav)
            prob = torch.sigmoid(logit).item()
            is_scream = prob >= threshold

            now = datetime.now()

            if detection == False and (t1 is None or abs((now - t1).total_seconds()) >= 5):
                detection = True
                print('탐지중...')

            if is_scream and detection:
                data = {}
                t1 = now
                detection = False
                time_str = now.strftime("%Y-%m-%d %H:%M:%S.%f").split(' ')

                data['detect_type'] = 0
                data['detect_date'] = time_str[0]
                data['detect_time'] = time_str[1]
                data['detect_zone'] = ''

                # JSON 문자열로 직렬화 후 바이트로 인코딩
                message = json.dumps(data).encode("utf-8")

                # print(message)

                sock.sendto(message, (UDP_IP, UDP_PORT))
                print("전송 완료")


            print(prob)

            # status = "🚨 SCREAM DETECTED" if is_scream else "… normal"
            # print(f"\rProb={prob:.3f}  =>  {status}           ", end="")

    except KeyboardInterrupt:
        print("\n[Realtime-Sliding] 종료합니다.")
    except Exception as e:
        print(f"\n[Realtime-Sliding] 오류 발생: {e}")

if __name__ == "__main__":
    # 학습시
    # main()

    # 실시간 테스트시
    run_realtime_scream_detection_sliding(
        checkpoint_path='./check_point/panns_scream_best_1.pt',
        sample_rate=32000,
        window_duration=2.0,
        hop_duration=0.5,
        threshold=0.7,
    )