import torch
import os
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_read
from audiocraft.data.audio_utils import convert_audio

musicgen = None
MUSICGEN_MODEL = 'facebook/musicgen-medium'

@torch.no_grad()
def seam_ce(paths, descriptions=None):
    global musicgen
    if musicgen is None:
        musicgen = MusicGen.get_pretrained(MUSICGEN_MODEL, device="cuda")
        
    if type(paths) == list:
        if type(descriptions) != list:
            descriptions = [descriptions] * len(paths)
        results = []
        for path, description in zip(paths, descriptions):
            results.append(seam_ce(path, description))
            
        return torch.stack(results)
    
    sample, sr = audio_read(paths)
    if len(sample.shape) < 2:
        sample = sample[None]
    sample = convert_audio(sample, sr, musicgen.sample_rate, musicgen.audio_channels)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        attributes, samples = musicgen._prepare_tokens_and_attributes(descriptions=[descriptions], prompt=sample[None])
        logs = musicgen.lm.compute_predictions(samples, conditions=attributes).logits

    logs = torch.nn.functional.cross_entropy(logs.permute(0, 3, 1, 2).float(), samples, reduction='none')[0, 0].cpu()
    
    return logs[..., logs.shape[-1]//2:][..., :5].mean()

def seam_perplexity(paths, descriptions=None):
    return torch.exp(seam_ce(paths, descriptions).mean())

def seam_perplexity_err(paths, descriptions=None):
    assert type(paths) == list, "paths should be a list"
    values = seam_ce(paths, descriptions)
    mean = values.mean()
    ci = 1.96 * values.std() * (values.shape[-1] ** -0.5)
    minimum = torch.exp(mean - ci)
    maximum = torch.exp(mean + ci)
    middle = 0.5 * maximum + 0.5 * minimum
    diff = (maximum - minimum) * 0.5
    return middle, diff

if __name__ == "__main__":

    samples = os.listdir(".")
    samples = [os.path.join("./", sample) for sample in filter(lambda x: x.endswith(".wav"), samples)]
    
    print("Seam Perplexity")
    print("Mean\t\tCI")
    mean, ci = seam_perplexity_err(samples)
    print(f"{mean:.2f}\t\t{ci:.2f}")