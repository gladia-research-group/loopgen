from beat_this.inference import File2Beats
import torch
import numpy as np
import gc
import os
from audiocraft.audiocraft.models import MusicGen
from src.loopgen import LoopGen
from audiocraft.audiocraft.data.audio import audio_write, audio_read
import random

FILE2BEATS_CKPT = "final0.ckpt"
MUSICGEN_MODEL = 'facebook/musicgen-medium'
MAGNET_MODEL = 'facebook/magnet-medium-30secs'

file2beats = None
musicgen = None
loopgen = None

MAX_CFG = 5.0
RESCORE_WEIGHT = 0.5

def get_beat_perfect_hint(path, min_length=5, max_length=10, unit_size=0.02, prompt_retries=3, length_retries=2):
    global file2beats
    if file2beats is None:
        file2beats = File2Beats(checkpoint_path=FILE2BEATS_CKPT, device="cuda", dbn=False)
    
    min_length = min_length // unit_size
    max_length = max_length // unit_size

    beats, downbeats = file2beats(path)

    diffs = []
    prev = None
    beats_per_bar = []
    beats_per_this_bar = 0
    downbeat_idx = 0
    for beat in beats:
        while downbeat_idx < len(downbeats) and beat > downbeats[downbeat_idx]:
            beats_per_bar.append(beats_per_this_bar)
            beats_per_this_bar = 0
            downbeat_idx += 1
        beats_per_this_bar += 1
        if prev is not None:
            diffs.append(beat - prev)
        prev = beat
        
    if len(beats_per_bar) < 1:
        return None

    diffs = sorted(diffs)
    beats_per_bar = sorted(beats_per_bar)

    beat_length = diffs[len(diffs)//2] // unit_size + 1 # Quantize to 0.02 the median
    beats_per_bar = beats_per_bar[len(beats_per_bar)//2] # Take the median

    bar_length = beats_per_bar * beat_length # Take what should be the length of a bar... based on the medians

    length = bar_length * 12
    assert length > 0, "Something went wrong"

    loops_done = 0
    while length < min_length or length > max_length:
        if length < min_length:
            length *= 2
        else:
            length /= 2
        loops_done += 1
    
    if loops_done > length_retries:
        return None

    return length


def musicgen_base(name, description, seed=42, reseed=True, duration=10.0):
    global musicgen
        
    path = f"musicgen_base_{name}_{seed}"
    if os.path.exists(path + ".wav"):
        a, sr = audio_read(path + ".wav")
        return a, sr, path + ".wav"
    
    if musicgen is None:
        musicgen = MusicGen.get_pretrained(MUSICGEN_MODEL, device="cuda")
    
    if reseed:
        torch.manual_seed(seed)
        np.random.seed(seed)

    musicgen.set_generation_params(
        duration = duration
    )
    
    result = musicgen.generate([description])[0].to(device="cpu", dtype=torch.float32)

    audio_write(path, result, musicgen.sample_rate, loudness_compressor=True, strategy="loudness", loudness_headroom_db=16)
    
    return result, musicgen.sample_rate, path + ".wav"


def musicgen_dumb_loop(name, description, seed=42):
    global musicgen
    
    result, sr, _ = musicgen_base(name, description, seed=seed)
    
    musicgen = None
    
    audio_write(f"musicgen_dumb_loop_{name}_{seed}", torch.cat([result, result], -1), sr, loudness_compressor=True, strategy="loudness", loudness_headroom_db=16)


def musicgen_beat_perfect_dumb_loop(name, description, seed=42, duration=10.0):
    global file2beats, musicgen
    torch.manual_seed(seed)
    np.random.seed(seed)
    tries = 0
    unit_length = None
    hint = None
    while unit_length is None:
        gc.collect()
        torch.cuda.empty_cache()
        if tries >= 5:
            unit_length = 350 * 3
            break
        del hint
        tries += 1
        hint, sr, path = musicgen_base(name, description, seed=seed, reseed=False, duration=duration)
        unit_length = get_beat_perfect_hint(path, 15, 30)
        
    file2beats = None
    
    hint = hint[..., :int(0.02 * sr) * int(unit_length)]
    
    audio_write(f"musicgen_beat_perfect_dumb_loop_{name}_{seed}", torch.cat([hint, hint], -1), sr, loudness_compressor=True, strategy="loudness", loudness_headroom_db=16)


def magnet_base(name, description, seed=42, reseed=True):
    global loopgen
    if loopgen is None:
        loopgen=LoopGen.get_pretrained(MAGNET_MODEL, device="cuda")
    
    path = f"magnet_base_{name}_{seed}"
    if os.path.exists(path + ".wav"):
        return audio_read(path + ".wav")[0], loopgen.sample_rate, path + ".wav"
    
    if reseed:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    setup = {
        'span_arrangement' : 'stride1',
        'use_sampling' : True,
        'top_k' : 0,
        'top_p' : .9,
        'temperature' : 3.0,
        'max_cfg_coef' : MAX_CFG,
        'min_cfg_coef' : 1.0,
        'decoding_steps' : [100, 50, 10, 10],
        'rescorer' : None,
        'rescore_weights' : 0.0
    }

    loopgen.set_generation_params(
        **setup
    )
    
    n_tokens = int(loopgen.duration * loopgen.frame_rate)
    print(n_tokens)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        results = loopgen.generate(text_prompt=[description], negative_text_prompt = None, valid_tokens=[n_tokens])
            
    results = results[0].to(device="cpu", dtype=torch.float32)

    audio_write(path, results, loopgen.sample_rate, loudness_compressor=True, strategy="loudness", loudness_headroom_db=16)

    return results, loopgen.sample_rate, path + ".wav"


def magnet_dumb_loop(name, description, seed=42):
    global musicgen
    
    result, sr, _ = magnet_base(name, description, seed=seed)
    
    musicgen = None
    
    audio_write(f"magnet_dumb_loop_{name}_{seed}", torch.cat([result, result], -1), sr, loudness_compressor=True, strategy="loudness", loudness_headroom_db=16)
    

def magnet_beat_perfect_dumb_loop(name, description, seed=42):
    global file2beats, loopgen
    torch.manual_seed(seed)
    np.random.seed(seed)
    unit_length = None
    tries = 0
    while unit_length is None:
        if tries >= 5:
            unit_length = 350 * 3
            break
        tries += 1
        hint, sr, path = magnet_base(name, description, seed=seed, reseed=False)
        unit_length = get_beat_perfect_hint(path, 15, 30)
        
    file2beats = None
    
    sr = loopgen.sample_rate
    hint = hint[..., :int(0.02 * sr) * int(unit_length)]
    
    audio_write(f"magnet_beat_perfect_dumb_loop_{name}_{seed}", torch.cat([hint, hint], -1), sr, loudness_compressor=True, strategy="loudness", loudness_headroom_db=16)


def magnet_loop(name, description, valid_tokens, hint=None, hint_sr=None, seed=42, prefix=""):
    global loopgen, musicgen

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if loopgen is None:
        loopgen=LoopGen.get_pretrained(MAGNET_MODEL, device="cuda")
    
    if musicgen is None:
        musicgen = MusicGen.get_pretrained(MUSICGEN_MODEL, device="cuda")
        
    max_duration  = int(loopgen.duration * loopgen.frame_rate)
    assert len(list(filter(lambda x: x <= max_duration, valid_tokens))) == len(valid_tokens), print(valid_tokens)
        
    setup = {
        'span_arrangement' : 'stride1',
        'use_sampling' : True,
        'top_k' : 0,
        'top_p' : .9,
        'temperature' : 3.0,
        'max_cfg_coef' : MAX_CFG,
        'min_cfg_coef' : 1.0,
        'decoding_steps' : [100, 50, 10, 10],
        'rescorer' : musicgen.lm,
        'rescore_weights' : RESCORE_WEIGHT,
        'offset': hint is None
    }

    loopgen.set_generation_params(
        **setup
    )

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        if hint is None:
            results = loopgen.generate(text_prompt=[description], negative_text_prompt = None, valid_tokens=valid_tokens)
        else:
            results = loopgen.generate_continuation(text_prompt=[description], negative_text_prompt = None, left_hint=hint, left_hint_sr=hint_sr, valid_tokens=valid_tokens)
            
    results = results[0].to(device="cpu", dtype=torch.float32)
    if valid_tokens[0] == max_duration:
        results = torch.cat([results, results], -1)
        

    audio_write(f"magnet_{prefix}_loop_{name}_{seed}", results, loopgen.sample_rate, loudness_compressor=True, strategy="loudness", loudness_headroom_db=16)
    

def magnet_hybrid_loop(name, description, valid_tokens=400, seed=42, prefix=""):
    global musicgen
    
    hint, sr, _ = musicgen_base(name, description, seed=seed)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    hint_len = int(valid_tokens * 640 / 32000 / 2 * sr) # Make hint half the valid tokens length
    hint = hint[..., :hint_len]
    
    magnet_loop(name, description, [valid_tokens], hint, sr, seed=seed, prefix=f"hybrid_{prefix}")
    
    
def magnet_beat_perfect_loop(name, description, seed=42, prefix=""):
    global file2beats, musicgen
    torch.manual_seed(seed)
    np.random.seed(seed)
    unit_length = None
    hint = None
    tries = 0
    while unit_length is None:
        gc.collect()
        torch.cuda.empty_cache()
        if tries >= 5:
            unit_length = 350 * 3
            break
        del hint
        tries += 1
        hint, sr, path = musicgen_base(name, description, seed=seed, reseed=False, duration=30.0)
        unit_length = get_beat_perfect_hint(path, 15, 30)
            
    file2beats = None
    gc.collect()
    torch.cuda.empty_cache()
    
    hint = [hint[..., :int(0.02 * sr) * int(unit_length / 2)]]
    valid_tokens = int(unit_length)
    
    magnet_loop(name, description, [valid_tokens], hint, sr, seed=seed, prefix="beat_perfect" + ("" if len(prefix)==0 else "_"+prefix))

if __name__ == "__main__":

    magnet_beat_perfect_loop("example", "A retro 80s synthwave track with analog synth arpeggios and punchy drums")
    gc.collect()
    torch.cuda.empty_cache()