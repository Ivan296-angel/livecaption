import numpy as np
import torch
from scipy.signal import resample_poly
from silero_vad import load_silero_vad, get_speech_timestamps
from math import gcd

class StreamingCheckpointBuffer:
    """Mono float32 sample-accurate segmentation. Call push() sequentially."""
    def __init__(self, sample_rate=48000, target_seconds=30.0,
                 min_seconds=25.0, max_seconds=45.0,
                 min_silence_ms=300, analysis_interval=1.0):
        if not (0 < min_seconds <= target_seconds <= max_seconds):
            raise ValueError('Expected 0 < min <= target <= max')
        self.sr = sample_rate
        self.target = round(target_seconds * sample_rate)
        self.minimum = round(min_seconds * sample_rate)
        self.maximum = round(max_seconds * sample_rate)
        self.interval = max(1, round(analysis_interval * sample_rate))
        self.min_silence_ms = min_silence_ms
        self.buffer = np.empty(0, dtype=np.float32)
        self.buffer_start = 0
        self.checkpoint_id = 0
        self.next_analysis = self.target
        self.vad_model = load_silero_vad()

    def find_silence(self, audio):
        divisor = gcd(self.sr, 16000)
        a16 = resample_poly(audio, 16000 // divisor,
                            self.sr // divisor).astype(np.float32)
        speech = get_speech_timestamps(
            torch.from_numpy(np.ascontiguousarray(a16)), self.vad_model,
            sampling_rate=16000, threshold=0.5,
            min_speech_duration_ms=150, min_silence_duration_ms=300,
            speech_pad_ms=0)
        intervals = []
        previous_end = 0
        for item in speech:
            begin = max(0, round(item['start'] * self.sr / 16000))
            end = min(len(audio), round(item['end'] * self.sr / 16000))
            if begin > previous_end and (begin - previous_end) * 1000 / self.sr >= self.min_silence_ms:
                intervals.append((previous_end, begin))
            previous_end = max(previous_end, end)
        if len(audio) > previous_end and (len(audio) - previous_end) * 1000 / self.sr >= self.min_silence_ms:
            intervals.append((previous_end, len(audio)))
        return intervals

    def choose_cut(self):
        silence = self.find_silence(self.buffer[:self.maximum])
        candidates = []
        for start, end in silence:
            left = max(start, self.minimum)
            right = min(end, self.maximum)
            if left < right:
                candidates.append(min(max(self.target, left), right))
        return min(candidates, key=lambda x: abs(x - self.target)) if candidates else None

    def emit_checkpoint(self, cut, boundary_type):
        start = self.buffer_start
        end = start + cut
        audio = self.buffer[:cut].copy()
        self.buffer = self.buffer[cut:].copy()
        self.buffer_start = end
        self.checkpoint_id += 1
        self.next_analysis = self.target
        return {'id': self.checkpoint_id, 'start_sample': start,
                'end_sample': end, 'start_seconds': start / self.sr,
                'end_seconds': end / self.sr, 'duration': len(audio) / self.sr,
                'boundary_type': boundary_type, 'audio': audio}

    def push(self, block):
        block = np.asarray(block, dtype=np.float32)
        if block.ndim != 1:
            raise ValueError('Expected mono audio')
        if block.size == 0:
            return []
        self.buffer = np.concatenate((self.buffer, block))
        completed = []
        while len(self.buffer) >= self.target:
            cut = None
            kind = None
            if len(self.buffer) >= self.next_analysis:
                cut = self.choose_cut()
                self.next_analysis = len(self.buffer) + self.interval
                if cut is not None:
                    kind = 'silence'
            if cut is None and len(self.buffer) >= self.maximum:
                cut, kind = self.maximum, 'forced'
            if cut is None:
                break
            completed.append(self.emit_checkpoint(cut, kind))
        return completed

    def flush(self):
        return ([self.emit_checkpoint(len(self.buffer), 'end')]
                if len(self.buffer) else [])
