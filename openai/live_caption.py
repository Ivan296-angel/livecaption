"""Step 30.8: GPU bridge scheduling with early checkpoint-relative LIVE captions with audio freshness timing.

Run from D:\\Work\\CaptionST after copying this file, caption_ui.py,
and streaming_checkpoint.py into that project directory.
"""
import threading
import time
from collections import deque
from queue import Queue

import numpy as np
import soundcard as sc
import torch
from faster_whisper import WhisperModel
from silero_vad import VADIterator, load_silero_vad, get_speech_timestamps

from caption_freshness import audio_clock
from caption_ui import CaptionUI
from answer_worker import AnswerWorker
from pathlib import Path
from streaming_checkpoint import StreamingCheckpointBuffer

MODEL_DIR = r'..\models\distil-medium.en'
SAMPLE_RATE = 48000
BLOCK_SECONDS = 0.1
BLOCK_FRAMES = round(SAMPLE_RATE * BLOCK_SECONDS)
VAD_SAMPLE_RATE = 16000
VAD_FRAME_SIZE = 512
PRE_ROLL_SECONDS = 1.0
END_SILENCE_MS = 900
LIVE_UPDATE_SECONDS = 1.0
# Experimental bridge scheduling: an early LIVE update may run ahead of an
# interim FINAL, but the FINAL wait is strictly bounded.
BRIDGE_WAIT_SECONDS = 1.0
BRIDGE_FIRST_UPDATE_SECONDS = 0.7
# Retain only audio since the last queued FINAL checkpoint. No 60s truncation.
# The checkpoint worker generally finalizes a segment every 25-45 seconds.
MIN_AUDIO_RMS = 0.008
MIN_SPEECH_SECONDS = 0.35

torch.set_num_threads(1)
stop_event = threading.Event()
gpu_ready = threading.Event()
condition = threading.Condition()
final_jobs = deque()  # (phrase_id, end_sample, audio_or_None, is_last, boundary_type, deadline, bridged)
latest_live_job = None
closed_phrase_ids = set()
active_phrase_id = None
committed_frames = {}  # audio whose FINAL inference has completed
live_cut_frames = {}  # audio assigned to queued FINAL (early cut)
# The audio capture thread prunes blocks after confirmation. This is separate
# from the checkpoint worker buffer so FINAL never depends on LIVE snapshots.
worker_stopping = False
confirmed_captions = []
# FINAL-only checkpoint audit. Never update this from LIVE hypotheses.
previous_final_checkpoint = {}


def audit_final_checkpoint(phrase_id, start_sample, end_sample, text, elapsed, boundary_type):
    previous = previous_final_checkpoint.get(phrase_id)
    print("\n" + "=" * 55)
    print(f"FINAL CHECKPOINT AUDIT | Phrase #{phrase_id}")
    print(f"Audio: {start_sample / SAMPLE_RATE:.2f}s - "
          f"{end_sample / SAMPLE_RATE:.2f}s (phrase-relative)")
    print(f"GPU inference: {elapsed:.2f}s")
    print(f"Boundary type: {boundary_type}")
    if previous is not None:
        previous_end, previous_text = previous
        print("Contiguous audio:", "YES" if previous_end == start_sample else "NO")
        print("PREVIOUS FINAL END:", " ".join(previous_text.split()[-12:]))
    print("CURRENT FINAL START:", " ".join(text.split()[:12]))
    print("=" * 55)
    previous_final_checkpoint[phrase_id] = (end_sample, text)


checkpoint_events = Queue()  # capture thread -> checkpoint worker
ui = CaptionUI(on_close=stop_event.set)

# Load candidate-supplied background only, never fabricate personal experience.
profile_path = Path(__file__).with_name('candidate_profile.txt')
profile_text = profile_path.read_text(encoding='utf-8') if profile_path.is_file() else ''


def on_answer_result(generation, status, text, elapsed):
    ui.send('answer_result', text=text, timing={
        'generation': generation, 'status': status, 'elapsed': elapsed})


def on_answer_chunk(generation, chunk, elapsed):
    ui.send('answer_chunk', text=chunk, timing={
        'generation': generation, 'elapsed': elapsed})


answer_worker = AnswerWorker(on_answer_result, profile=profile_text,
                             on_chunk=on_answer_chunk)


def on_question(event):
    print(f"[ANSWER ROUTE] source={event.source} phrase={event.phrase_id} "
          f"chars={len(event.text)}", flush=True)
    generation = answer_worker.submit(event.text)
    if generation is None:
        raise RuntimeError("Answer worker rejected the question (worker stopped or empty text).")
    ui.send('answer_started', timing={'generation': generation})


ui.on_question = on_question
print(f"[ANSWER WIRING] Connected | app={Path(__file__).resolve()} "
      f"| UI={Path(__import__('caption_ui').__file__).resolve()} "
      f"| worker={Path(__import__('answer_worker').__file__).resolve()}", flush=True)


def transcribe(model, audio):
    # Correct for 48 kHz input; production can replace this with resample_poly.
    audio16 = np.ascontiguousarray(audio[::3], dtype=np.float32)
    start = time.perf_counter()
    segments, _ = model.transcribe(
        audio16, language='en', task='transcribe', beam_size=1,
        best_of=1, temperature=0.0, vad_filter=False,
        condition_on_previous_text=False, without_timestamps=True,
        no_speech_threshold=0.6, log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4)
    text = ' '.join(s.text.strip() for s in segments).strip()
    return text, time.perf_counter() - start


def submit_live(phrase_id, start_sample, audio, captured_at, endpoint_frame):
    """Replace pending LIVE job. Tag it with its phrase-relative audio start."""
    global latest_live_job
    with condition:
        if phrase_id in closed_phrase_ids or worker_stopping:
            return
        # A checkpoint may have finalized while this snapshot was prepared.
        # We must not show text that begins inside already-confirmed audio.
        if start_sample < live_cut_frames.get(phrase_id, 0):
            return
        latest_live_job = (phrase_id, start_sample, audio, time.perf_counter(), captured_at, endpoint_frame)
        condition.notify()


def close_live(phrase_id):
    global latest_live_job, active_phrase_id
    with condition:
        closed_phrase_ids.add(phrase_id)
        if latest_live_job is not None and latest_live_job[0] == phrase_id:
            latest_live_job = None
        if active_phrase_id == phrase_id:
            active_phrase_id = None


def submit_final(phrase_id, checkpoint, last=False):
    """Queue FINAL and immediately reset the LIVE audio starting boundary.

    The FINAL transcript is appended only after the GPU returns. At submission
    we instead freeze the old LIVE hypothesis until that FINAL arrives.
    """
    global latest_live_job
    audio = checkpoint['audio'] if checkpoint is not None else None
    end_sample = checkpoint['end_sample'] if checkpoint is not None else None
    boundary_type = checkpoint['boundary_type'] if checkpoint is not None else 'empty'
    with condition:
        if audio is not None and end_sample is not None:
            previous_cut = live_cut_frames.get(phrase_id, 0)
            if end_sample <= previous_cut:
                raise RuntimeError('Checkpoint boundary must increase')
            live_cut_frames[phrase_id] = end_sample
            # Invalidate an older pending snapshot; running snapshots will be
            # discarded before UI emission by the GPU worker.
            if (latest_live_job is not None
                    and latest_live_job[0] == phrase_id
                    and latest_live_job[1] < end_sample):
                latest_live_job = None
        # Send split before the GPU is allowed to send its FINAL result,
        # guaranteeing UI event order even on a very fast GPU.
        if audio is not None and end_sample is not None and not last:
            ui.send('split', phrase_id, timing={'end_sample': end_sample})
            print(f'[LIVE EARLY CUT #{phrase_id}] '
                  f'Checkpoint queued through {end_sample/SAMPLE_RATE:.2f}s')
        # When speech continues, allow at most one *newer* LIVE hypothesis to
        # run before this FINAL checkpoint. Tail/last FINAL never waits.
        deadline = (time.perf_counter() + BRIDGE_WAIT_SECONDS
                    if not last and audio is not None else 0.0)
        final_jobs.append((phrase_id, end_sample, audio, last,
                           boundary_type, deadline, False))
        condition.notify()


def gpu_worker():
    global latest_live_job
    try:
        ui.send('status', text='Loading GPU Whisper...')
        model = WhisperModel(MODEL_DIR, device='cuda', device_index=0,
                             compute_type='int8_float32', local_files_only=True)
        gpu_ready.set()
        ui.send('status', text='GPU Whisper ready')
        while True:
            with condition:
                # Do not postpone an interim FINAL indefinitely. The bridge
                # deadline is evaluated on a monotonic clock inside the lock.
                while True:
                    if final_jobs:
                        (fid, fend, faudio, flast, ftype,
                         deadline, bridged) = final_jobs[0]
                        now = time.perf_counter()
                        valid_bridge = (
                            not flast and not bridged and not worker_stopping
                            and now < deadline
                            and latest_live_job is not None
                            and latest_live_job[0] == fid
                            and latest_live_job[1] >= fend
                            and fid not in closed_phrase_ids
                            and fid == active_phrase_id
                        )
                        if valid_bridge:
                            # Reserve the bridge slot *before* releasing lock.
                            # FINAL remains at the head and is selected next.
                            final_jobs[0] = (
                                fid, fend, faudio, flast, ftype, deadline, True
                            )
                            (phrase_id, live_start, audio, submitted_at,
                             captured_at, endpoint_frame) = latest_live_job
                            latest_live_job = None
                            mode = 'live'
                            print(f'[GPU BRIDGE #{fid}] LIVE before FINAL '
                                  f'ending at {fend/SAMPLE_RATE:.2f}s')
                            break

                        if (not flast and not bridged
                                and not worker_stopping and now < deadline):
                            # Leave the GPU idle briefly for a fresh LIVE job.
                            # This is a latency trade-off: FINAL may be delayed
                            # by <= BRIDGE_WAIT_SECONDS once per checkpoint.
                            condition.wait(timeout=deadline-now)
                            continue

                        (phrase_id, end_sample, audio, last,
                         boundary_type, _, _) = final_jobs.popleft()
                        mode = 'final'
                        break
                    if latest_live_job is not None:
                        (phrase_id, live_start, audio, submitted_at,
                         captured_at, endpoint_frame) = latest_live_job
                        latest_live_job = None
                        mode = 'live'
                        break
                    if worker_stopping:
                        return
                    condition.wait()
            try:
                if mode == 'final' and audio is None:
                    text, elapsed = '', 0.0
                else:
                    started_at = time.perf_counter()
                    text, elapsed = transcribe(model, audio)
                with condition:
                    if mode == 'live':
                        if phrase_id in closed_phrase_ids or phrase_id != active_phrase_id:
                            continue
                        if live_start < live_cut_frames.get(phrase_id, 0):
                            # Checkpoint finalized while this LIVE job was running.
                            continue
                        if text:
                            # Record timings with a monotonic clock shared by all threads.
                            # Tkinter measures the remaining UI queue/render delay.
                            finished_at = time.perf_counter()
                            ui.send('live', phrase_id, text, timing={
                                'captured_at': captured_at,
                                'endpoint_frame': endpoint_frame,
                                'sample_rate': SAMPLE_RATE,
                                'submitted_at': submitted_at,
                                'started_at': started_at,
                                'finished_at': finished_at,
                                'audio_seconds': len(audio) / SAMPLE_RATE,
                                'phrase_id': phrase_id,
                                'live_start_sample': live_start,
                            })
                        else:
                            print(f'[LIVE PERF] Empty hypothesis | GPU={elapsed:.2f}s')
                    else:
                        if text:
                            confirmed_captions.append(text)
                        if end_sample is not None and audio is not None:
                            audit_final_checkpoint(
                                phrase_id,
                                end_sample - len(audio),
                                end_sample,
                                text,
                                elapsed,
                                boundary_type,
                            )
                        if end_sample is not None and audio is not None:
                            # Confirm audio even when Whisper produces empty text.
                            # Otherwise LIVE keeps retranscribing already processed audio.
                            committed_frames[phrase_id] = end_sample
                            # Invalidate a pending pre-checkpoint LIVE snapshot.
                            if (latest_live_job is not None
                                    and latest_live_job[0] == phrase_id
                                    and latest_live_job[1] < end_sample):
                                latest_live_job = None
                            print(f'[LIVE RESET #{phrase_id}] '
                                  f'Confirmed through {end_sample/SAMPLE_RATE:.2f}s')
                        # Interim checkpoint appends FINAL without closing phrase.
                        ui.send('final' if last else 'checkpoint', phrase_id, text,
                                timing={'end_sample': end_sample})
                        if last:
                            previous_final_checkpoint.pop(phrase_id, None)
                print(f'[{mode.upper()} #{phrase_id}] {elapsed:.2f}s')
            except Exception as exc:
                ui.send('error', text=f'Whisper error: {exc}')
                if mode == 'final' and last:
                    ui.send('final', phrase_id, '')
    except Exception as exc:
        ui.send('error', text=f'GPU initialization error: {exc}')
        stop_event.set()
    finally:
        gpu_ready.set()
        print('[GPU] Worker stopped')


def valid_audio(audio, model):
    if len(audio) < int(0.4 * SAMPLE_RATE):
        return False
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms < MIN_AUDIO_RMS:
        return False
    a16 = np.ascontiguousarray(audio[::3], dtype=np.float32)
    segments = get_speech_timestamps(
        torch.from_numpy(a16), model, sampling_rate=16000,
        threshold=0.55, min_speech_duration_ms=200)
    duration = sum((s['end'] - s['start']) / 16000 for s in segments)
    return duration >= MIN_SPEECH_SECONDS


def checkpoint_worker():
    """Does heavy checkpoint VAD work outside the WASAPI capture thread."""
    current_id = None
    checkpoint = None
    model = load_silero_vad()  # distinct from streaming VADIterator
    emitted_any = False
    try:
        while True:
            kind, phrase_id, payload = checkpoint_events.get()
            try:
                if kind == 'stop':
                    break
                if kind == 'start':
                    if checkpoint is not None:
                        raise RuntimeError('Start arrived before previous phrase ended')
                    current_id = phrase_id
                    checkpoint = StreamingCheckpointBuffer(sample_rate=SAMPLE_RATE)
                    emitted_any = False
                    chunks = checkpoint.push(payload)
                elif kind == 'audio':
                    if current_id != phrase_id or checkpoint is None:
                        raise RuntimeError('Checkpoint audio out of order')
                    chunks = checkpoint.push(payload)
                elif kind == 'end':
                    if current_id != phrase_id or checkpoint is None:
                        raise RuntimeError('Checkpoint end out of order')
                    chunks = checkpoint.flush()
                    # Empty chunks are possible only on exact boundaries.
                    if not chunks:
                        submit_final(phrase_id, None, last=True)
                    else:
                        tail = chunks[-1]
                        for part in chunks[:-1]:
                            submit_final(phrase_id, part, last=False)
                            emitted_any = True
                        if valid_audio(tail['audio'], model):
                            submit_final(phrase_id, tail, last=True)
                        else:
                            print(f'[Rejected tail #{phrase_id}]')
                            submit_final(phrase_id, None, last=True)
                    current_id = None
                    checkpoint = None
                    continue
                else:
                    raise RuntimeError(f'Unknown checkpoint event {kind}')

                for part in chunks:
                    emitted_any = True
                    print(f"[Checkpoint #{phrase_id}] "
                          f"{part['start_seconds']:.2f}-{part['end_seconds']:.2f}s "
                          f"({part['boundary_type']})")
                    submit_final(phrase_id, part, last=False)
            except Exception as exc:
                ui.send('error', text=f'Checkpoint error: {exc}')
                stop_event.set()
                raise
            finally:
                checkpoint_events.task_done()
    except Exception as exc:
        print('[Checkpoint error]', exc)
    finally:
        print('[Checkpoint] Worker stopped')


def capture_worker():
    global active_phrase_id
    phrase_id = 0
    captured_frames_total = 0
    speaking = False
    live_elapsed = 0.0
    last_seen_live_cut = 0
    bridge_first_update = False
    # Entries: (phrase_relative_start_frame, audio_chunk)
    # No maxlen: prune audio when it has been assigned to a queued FINAL.
    live_blocks = deque()
    live_frames = 0
    history = deque(maxlen=max(1, round(PRE_ROLL_SECONDS / BLOCK_SECONDS)))
    vad_buffer = np.empty(0, dtype=np.float32)

    def send_end():
        nonlocal speaking, live_elapsed, live_frames, bridge_first_update, last_seen_live_cut
        if not speaking:
            return
        close_live(phrase_id)
        ui.send('end', phrase_id)
        checkpoint_events.put(('end', phrase_id, None))
        speaking = False
        live_elapsed = 0.0
        bridge_first_update = False
        last_seen_live_cut = 0
        live_frames = 0
        live_blocks.clear()

    try:
        streaming_vad = load_silero_vad()
        vad = VADIterator(
            streaming_vad, threshold=0.5, sampling_rate=VAD_SAMPLE_RATE,
            min_silence_duration_ms=END_SILENCE_MS, speech_pad_ms=200)
        vad.reset_states()
        speaker = sc.default_speaker()
        if speaker is None:
            raise RuntimeError('No default playback device found')
        loopback = next((m for m in sc.all_microphones(include_loopback=True)
                         if m.isloopback and m.id == speaker.id), None)
        if loopback is None:
            raise RuntimeError(f'Loopback not found for {speaker.name}')
        gpu_ready.wait()
        if stop_event.is_set():
            return
        ui.send('status', text=f'Listening: {speaker.name}')
        print(f'[Audio] Capturing: {speaker.name}')

        with loopback.recorder(samplerate=SAMPLE_RATE,
                               blocksize=BLOCK_FRAMES * 4) as recorder:
            while not stop_event.is_set():
                block = recorder.record(numframes=BLOCK_FRAMES)
                # Timestamp when the newest audio block becomes available to Python.
                # This is NOT the exact time the last spoken word occurred.
                captured_at = time.perf_counter()
                captured_frames_total += len(block)
                audio_clock.update(captured_frames_total, captured_at)
                if block.ndim == 2:
                    block = block.mean(axis=1)
                block = np.asarray(block, dtype=np.float32)
                vad_buffer = np.concatenate((vad_buffer, block[::3]))
                started = ended = False
                while len(vad_buffer) >= VAD_FRAME_SIZE:
                    frame = np.ascontiguousarray(vad_buffer[:VAD_FRAME_SIZE])
                    vad_buffer = vad_buffer[VAD_FRAME_SIZE:]
                    event = vad(torch.from_numpy(frame))
                    if event:
                        started |= 'start' in event
                        ended |= 'end' in event

                if started and not speaking:
                    speaking = True
                    phrase_id += 1
                    with condition:
                        active_phrase_id = phrase_id
                    initial = list(history) + [block.copy()]
                    history.clear()
                    live_blocks.clear()
                    live_frames = 0
                    for part in initial:
                        live_blocks.append((live_frames, part))
                        live_frames += len(part)
                    checkpoint_events.put(('start', phrase_id, np.concatenate(initial)))
                    live_elapsed = 0.0
                    bridge_first_update = False
                    last_seen_live_cut = 0
                    ui.send('start', phrase_id)
                elif speaking:
                    item = block.copy()
                    live_blocks.append((live_frames, item))
                    live_frames += len(item)
                    checkpoint_events.put(('audio', phrase_id, item))
                else:
                    history.append(block.copy())

                if speaking and not ended:
                    # Capture is not blocked by FINAL GPU work; check the cut
                    # cheaply and send the first new LIVE snapshot sooner.
                    with condition:
                        confirmed_until = live_cut_frames.get(phrase_id, 0)
                    if confirmed_until > last_seen_live_cut:
                        last_seen_live_cut = confirmed_until
                        live_elapsed = 0.0
                        bridge_first_update = True
                    live_elapsed += BLOCK_SECONDS
                    interval = (BRIDGE_FIRST_UPDATE_SECONDS
                                if bridge_first_update else LIVE_UPDATE_SECONDS)
                    if live_elapsed >= interval:
                        live_elapsed = 0.0
                        bridge_first_update = False
                        # Retire the queued checkpoint prefix, including a partial block.
                        while live_blocks and (live_blocks[0][0] + len(live_blocks[0][1]) <= confirmed_until):
                            live_blocks.popleft()
                        if live_blocks and live_blocks[0][0] < confirmed_until:
                            start, data = live_blocks.popleft()
                            live_blocks.appendleft((confirmed_until, data[confirmed_until-start:].copy()))

                        if live_blocks:
                            # Exact, contiguous audio after the queued checkpoint.
                            snapshot_start = live_blocks[0][0]
                            snapshot = np.concatenate([part for _, part in live_blocks])
                            if len(snapshot) >= round(0.4 * SAMPLE_RATE):
                                submit_live(phrase_id, snapshot_start, snapshot, captured_at, captured_frames_total)

                if ended and speaking:
                    send_end()
        send_end()
    except Exception as exc:
        ui.send('error', text=f'Capture error: {exc}')
    finally:
        # A mid-stream exception should still close an active phrase.
        send_end()
        stop_event.set()


def main():
    global latest_live_job, worker_stopping
    gpu_thread = threading.Thread(target=gpu_worker, name='WhisperGPU')
    checkpoint_thread = threading.Thread(target=checkpoint_worker, name='Checkpoints')
    capture_thread = threading.Thread(target=capture_worker, name='AudioCapture')
    gpu_thread.start()
    checkpoint_thread.start()
    capture_thread.start()

    def shutdown_monitor():
        global latest_live_job, worker_stopping
        stop_event.wait()
        capture_thread.join()
        checkpoint_events.put(('stop', None, None))
        checkpoint_thread.join()
        with condition:
            worker_stopping = True
            latest_live_job = None
            condition.notify_all()
        gpu_thread.join()
        answer_worker.close()
        ui.send('stopped')

    threading.Thread(target=shutdown_monitor,
                     name='ShutdownMonitor', daemon=True).start()
    try:
        ui.run()  # Tkinter only on main thread
    finally:
        stop_event.set()
        answer_worker.close()
        capture_thread.join()
        checkpoint_thread.join()
        gpu_thread.join()
    print('\nCOMPLETE FINAL TRANSCRIPT')
    print(' '.join(confirmed_captions))


if __name__ == '__main__':
    main()
