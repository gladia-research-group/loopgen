import torch
import numpy as np
from audiocraft.audiocraft.models import MusicGen
from audiocraft.audiocraft.data.audio import audio_read
from audiocraft.audiocraft.data.audio_utils import convert_audio
import tqdm
import os
import shutil
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from name_descriptions import SAMPLES_DESCRIPTIONS

import pickle

from fadtk.fad import FrechetAudioDistance
#from kadtk.kad import KernelAudioDistance
from fadtk.model_loader import *
from fadtk.fad_batch import cache_embedding_files

musicgen = None
MUSICGEN_MODEL = 'facebook/musicgen-medium'
PATH = "samples/"
SIZE = 1000

def get_tech_from_filename(filename):
    filename = filename[:-4]
    
    counter = 2
    for i in range(len(filename)-1, -1, -1):
        if filename[i] == "_":
            counter -= 1
        if counter == 0:
            break
    
    return filename[:i]

def get_name_from_filename(filename):
    filename = filename[:-4]
    
    return filename.split("_")[-2]


def all_techniques():
    all_files = os.listdir(PATH)
    techs = dict()
    for filename in all_files:
        if filename[-4:] != ".wav":
            continue        
        tech = get_tech_from_filename(filename)
        techs[tech] = techs.get(tech, 0) + 1
        
    return techs

def filter_techniques(all_techs, keep_base=False):
    filtered_techs = set()
    for k,v in all_techs.items():
        if v < 2:
            continue
        if "base" in k and not keep_base:
            continue
        filtered_techs.add(k)
        
    return filtered_techs

def get_files_per_tech(all_techs):
    all_files = os.listdir(PATH)
    files_per_tech = dict()
    for filename in all_files:
        tech = get_tech_from_filename(filename)
        if tech in all_techs:
            files_per_tech[tech] = files_per_tech.get(tech, []) + [filename]
            
    return files_per_tech

@torch.no_grad()
def calc_ce_on_file(filename):
    global musicgen
    if musicgen is None:
        musicgen = MusicGen.get_pretrained(MUSICGEN_MODEL, device="cuda")
    
    name = get_name_from_filename(filename)
    description = SAMPLES_DESCRIPTIONS[name]
    
    sample, sr = audio_read(PATH + filename)
    if len(sample.shape) < 2:
        sample = sample[None]
    sample = convert_audio(sample, sr, musicgen.sample_rate, musicgen.audio_channels)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        attributes, samples = musicgen._prepare_tokens_and_attributes(descriptions=[description], prompt=sample[None])
        logs = musicgen.lm.compute_predictions(samples, conditions=attributes).logits

    logs = torch.nn.functional.cross_entropy(logs.permute(0, 3, 1, 2).float(), samples, reduction='none')[0, 0].cpu()
    
    return logs


fad_models = {m.name: m for m in get_all_models()}

def calc_fad(model="vggish"):    
    model = fad_models[model]
    baseline = "fma_pop"
    scores_per_tech = dict()
    
    fad = FrechetAudioDistance(model, audio_load_worker=1, load_model=False)
    files_per_tech = get_files_per_tech(filter_techniques(all_techniques(), keep_base=True))
    for tech, filenames in files_per_tech.items():
        new_path = PATH + tech
        if os.path.exists(new_path):
            shutil.rmtree(new_path)
        os.mkdir(new_path)
        for filename in filenames:
            shutil.copyfile(PATH + filename, new_path + "/" + filename)
        cache_embedding_files(new_path, model, workers=1)
        scores_per_tech[tech] = fad.score(baseline, new_path + "/")
        gc.collect()
        torch.cuda.empty_cache()
        shutil.rmtree(new_path)
        
    return scores_per_tech

def make_plot(df):
    
    sns.set(
        style='white', context='talk', 
        rc={'axes.facecolor': (0, 0, 0, 0), 'figure.facecolor':"white"}
    )
    g = sns.FacetGrid(df, row="tech", hue=f"Mean si_score", aspect=14, height=.8, palette="crest")
    
    max_value = df["si_score"].max()
    

   # Draw the densities in a few steps
    # g.map(sns.histplot, score, clip_on=False, binwidth=.5 if score[:2] == 'sm' else 1.5,
    #     fill=True, alpha=.9, linewidth=1.5)
    # # g.map(sns.histplot, "sm_score", clip_on=False, color="w", lw=2, bins=20)
    
    g.map(
        sns.kdeplot, "si_score", clip_on=False,
        fill=True, alpha=.9, lw=0, bw_adjust=.2
    )
    g.map(
        sns.kdeplot, "si_score", clip_on=False,
        lw=3, bw_adjust=.2, color='white'
    )

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)


    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .16, x.iloc[0], fontweight="bold", color="black",
                ha="left", va="center", transform=ax.transAxes, size=18)
        ax.text(
            1, .2, f'{float(label):.2f}', fontweight='bold',
            color=color, ha='right', va='center', transform=ax.transAxes, size=22
        )
        ax.set_ylabel('')


    g.map(label, "tech")

    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-.4, bottom=.12)

    # Remove axes details that don't play well with overlap
    g.set_titles('')
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    
    first_ax = g.axes.flatten()[0]
    last_ax = g.axes.flatten()[-1]
    
    
    first_ax.text(
        1, .6, f'Mean', fontweight='bold', color='k',
        ha='right', va='baseline', transform=first_ax.transAxes, size=28
    )
    first_ax.text(
        0, .6, 'Model', fontweight='bold', color='k',
        ha='left', va='baseline', transform=first_ax.transAxes, size=28
    )
    
    last_ax.set_xlabel('\\textbf{Seam Perplexity}', fontsize=28)
    
    scale = np.asarray([1, 10, 100, 1000, 10000, 100000, 1000000])
    scale = scale[scale <= max_value]
    plt.xticks(scale, ["1e0", "1e1", "1e2", "1e3", "1e4", "1e5", "1e6"][:scale.shape[0]])
    
    plt.savefig(f'seamlessness_ridgeplot.pdf', bbox_inches = "tight")

if __name__ == "__main__":
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
        }
    )
    
    
    files_per_tech = get_files_per_tech(filter_techniques(all_techniques()))
    #sns.set_theme(style="darkgrid")
    #palette = sns.color_palette("deep", n_colors=len(files_per_tech), as_cmap=True)

    values_per_tech = dict()
    
    fig, ax = plt.subplots()
    
    logs_per_tech = dict()
    
    for tech, filenames in tqdm.tqdm(files_per_tech.items()):
        results = []
        for filename in filenames:
            results.append(calc_ce_on_file(filename))
        logs_per_tech[tech] = results
        
    with open("ces10sec.pickle",'wb') as handle:
        pickle.dump(logs_per_tech, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
    # df = {"tech": [], "si_score": [], "sm_score": [], "Mean sm_score": [], "Mean si_score": []}
    # ordered_tech = []
    # for tech in logs_per_tech.keys():
    #     n = 0
    #     if ("dumb" not in tech) and ("naive" not in tech):
    #         n += 1
    #     if "musicgen" not in tech:
    #         n += 1
    #     if "beat_perfect" in tech:
    #         n += 1
            
    #     tech = tech.replace("050", "50")
    #     tech = tech.replace("25", "025")
    #     tech = tech.replace("50", "050")
    #     tech = tech.replace("75", "075")
            
    #     ordered_tech.append(str(n) + tech)
        
    # for tech in sorted(ordered_tech):
    #     tech = tech[1:]
    #     tech = tech.replace("025", "25")
    #     tech = tech.replace("050", "50")
    #     tech = tech.replace("075", "75")
    #     if tech == "magnet_beat_perfect_resc_50_loop":
    #         results = logs_per_tech["magnet_beat_perfect_resc_050_loop"]
    #     else:
    #         results = logs_per_tech[tech]
    #     abs_maxes = []
    #     abs_wind = []
    #     sum_ces = torch.zeros(SIZE)
    #     tot = torch.zeros(SIZE)
    #     for sample in results:
    #         empty_space = (SIZE - sample.shape[-1])//2
    #         sum_ces[empty_space:empty_space+sample.shape[-1]] += sample
    #         tot[empty_space:empty_space+sample.shape[-1]] += 1
    #         sample = sample[sample.shape[-1]//2:]
            
    #         # sm_score
    #         abs_maxes.append(sample[:5].max())
            
    #         # si_score
    #         abs_wind.append(sample[:5].mean())
            
            
    #     values = dict()

    #     abs_maxes = torch.stack(abs_maxes)
    #     abs_wind = torch.stack(abs_wind)
        
    #     nice_name = tech.replace("dumb", "Naive").replace("_loop", "")
    #     nice_name = nice_name.replace("naive", "Naive")
    #     nice_name = nice_name.replace("_", " ")
    #     nice_name = nice_name.replace("cfg ", "\lambda=")
    #     nice_name = nice_name.replace("\lambda=5", "\lambda=5\ ")
    #     nice_name = nice_name.replace("\lambda=10", "\lambda=10\ ")
    #     nice_name = nice_name.replace("resc ", "\omega=")
    #     nice_name = nice_name.replace("25", "0.25")
    #     nice_name = nice_name.replace("50", "0.50")
    #     nice_name = nice_name.replace("75", "0.75")
    #     nice_name = nice_name.replace("100", "1.00")
    #     if "\lambda=5" in nice_name and "\omega" not in nice_name:
    #         nice_name += "\omega=0.00"
            
    #     nice_name = nice_name.replace("magnet", "\\textrm{\\texttt{MAGNeT}")
    #     nice_name = nice_name.replace("hybrid", "Hybrid")
    #     nice_name = nice_name.replace("musicgen", "\\textrm{\\texttt{MusicGen}")
    #     nice_name = nice_name.replace("beat", "Beat")
    #     nice_name = nice_name.replace("perfect", "Aligned")
    #     nice_name = nice_name.replace("masked", "Masked")
        
    #     if "Masked" in nice_name:
    #         continue
        
    #     if "\lambda" in nice_name:          
    #         nice_name = nice_name.replace("\lambda", "Tiled}\,\lambda")
    #         if "10" in nice_name:
    #             continue
    #         if "Hybrid" in nice_name:
    #             if "0.50" not in nice_name:
    #                 continue
    #             nice_name = "\\textrm{\\texttt{MAGNeT} Hybrid Tiled}"
    #         else:
    #             if "0.50" not in nice_name:
    #                 continue
    #             nice_name = "\\textrm{\\texttt{MAGNeT} Tiled}"
    #     elif "Beat Aligned" in nice_name and "Naive" not in nice_name:
    #         if "\omega" in nice_name:
    #             nice_name = "\\textrm{\\texttt{MAGNeT} Beat Aligned Tiled}"
    #         else:
    #             continue
    #     else:
    #         nice_name += "}"
            
    #     nice_name = "$" + nice_name + "$"
        
    #     df["tech"] += [nice_name] * abs_maxes.shape[0]
    #     df["si_score"] += abs_wind.tolist()
    #     df["sm_score"] += abs_maxes.tolist()
        
    #     ci = 1.96*abs_maxes.std()*(abs_maxes.shape[-1]**-.5)       
    #     df["Mean sm_score"] += [(.5 * (2.71 ** (abs_maxes.mean() - ci)) + .5 * (2.71 ** (abs_maxes.mean() + ci))).item()] * abs_maxes.shape[0]
        
    #     ci = 1.96*abs_wind.std()*(abs_wind.shape[-1]**-.5)        
    #     df["Mean si_score"] += [(.5 * (2.71 ** (abs_wind.mean() - ci)) + .5 * (2.71 ** (abs_wind.mean() + ci))).item()] * abs_maxes.shape[0]
        
    #     mean_ce = sum_ces / tot

    #     sns.lineplot(x=torch.linspace((mean_ce.shape[0] // 2  + 1)* -0.02,mean_ce.shape[0] // 2 * 0.02, mean_ce.shape[0]), y=mean_ce, alpha=.95, label=nice_name, ax=ax)
        
    #     values["sm_score"] = (abs_maxes.mean(), abs_maxes.std(), abs_maxes.shape[-1])
    #     values["si_score"] = (abs_wind.mean(), abs_wind.std(), abs_wind.shape[-1])
    #     values_per_tech[tech] = values
         
    # ax.axvline(0, 0, 11, ls="--", lw=1, c="black")
    # plt.xlabel("\\textbf{Time (s)}", fontsize=28)
    # plt.ylabel("\\textbf{Cross-Entropy}", fontsize=28)
    # plt.xlim(-1,1)
    # sns.move_legend(ax, "lower center", ncol=3,
    #     bbox_to_anchor=[0.5, -0.4],
    #     markerscale=1.5,
    #     frameon=False,
    #     labelspacing=1.5)
    # plt.setp(ax.get_legend().get_texts(), fontsize='22')
    # ax.tick_params(axis='both', which='major', labelsize=18)
    # fig.savefig('abs_ce.pdf', bbox_inches = "tight")
    
    # plt.clf()
    
    # df = pd.DataFrame(data=df)
    # df['Mean sm_score'] = df.groupby('tech')['Mean sm_score'].transform('mean')
    # df['Mean si_score'] = df.groupby('tech')['Mean si_score'].transform('mean')
    
    # # make_plot(df, "sm_score")
    # # plt.clf()
    
    # make_plot(df)
    
    # gc.collect()
    # torch.cuda.empty_cache()
    
    # fads = calc_fad()
    # for tech in fads.keys():
    #     values = values_per_tech.get(tech, dict())
    #     values["fad"] = fads[tech]
    #     values_per_tech[tech] = values
        
    # fads = calc_fad("clap-laion-music")
    # for tech in fads.keys():
    #     values = values_per_tech.get(tech, dict())
    #     values["fad_clap"] = fads[tech]
    #     values_per_tech[tech] = values
   
    # all_metrics = []
    # for name, results in values_per_tech.items():
    #     print(f"{name}:")
    #     for metric, value in results.items():
    #         if metric in ["fad_vggish", "fad_clap"]:
    #             print(f"\t{metric}: {value:.2f}")
    #             if metric not in all_metrics:
    #                 all_metrics.append(metric)
    #         else:
    #             ci = 1.96*value[1]*(value[2]**-.5)
    #             minimum = 2.71 ** (value[0] - ci)
    #             maximum = 2.71 ** (value[0] + ci)
    #             middle = .5 * maximum + .5 * minimum
    #             diff = (maximum - minimum) * .5
    #             print(f"\t{metric}: {middle:.2f}±{diff:.2f}")
    #             if metric not in all_metrics:
    #                 all_metrics.append(metric)
                    
    # data = dict()
    # for name, results in values_per_tech.items():
    #     data["tech"] = data.get("tech", []) + [name]
    #     added = [] 
    #     for metric, value in results.items():
    #         if metric in ["fad", "kad"]:
    #             data[metric] = data.get(metric, []) + [f"{value:.2f}"]
    #             added.append(metric)
    #         else:
    #             ci = 1.96*value[1]*(value[2]**-.5)
    #             minimum = 2.71 ** (value[0] - ci)
    #             maximum = 2.71 ** (value[0] + ci)
    #             middle = .5 * maximum + .5 * minimum
    #             diff = (maximum - minimum) * .5
    #             data[metric] = data.get(metric, []) + [f"{middle.item():.2f}"]
    #             data[metric + "_ci"] = data.get(metric + "_ci", []) + [f"{diff.item():.2f}"]
    #             added.append(metric)
        
    #     for metric in set(all_metrics) - set(added):
    #         data[metric] = data.get(metric, []) + [None]
    #         if metric not in ["fad", "kad"]:
    #             data[metric + "_ci"] = data.get(metric + "_ci", []) + [None]
                
    # df = pd.DataFrame(data=data)
    # df.to_csv("metrics.csv")