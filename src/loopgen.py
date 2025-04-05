# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Main model for using MAGNeT. This will combine all the required components
and provide easy access to the generation API.
"""
import typing as tp
import torch
import gc

from audiocraft.audiocraft.models.genmodel import BaseGenModel
from audiocraft.audiocraft.models.magnet import MAGNeT
from audiocraft.audiocraft.models.lm import LMModel
from audiocraft.audiocraft.modules.conditioners import ConditioningAttributes
from audiocraft.audiocraft.data.audio_utils import convert_audio
from audiocraft.audiocraft.models.loaders import load_compression_model, load_lm_model_magnet


class LoopGen(MAGNeT):
	"""MAGNeT main model with convenient generation API.
	Args:
			See MusicGen class.
	"""
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	@staticmethod
	def get_pretrained(name: str = 'facebook/magnet-small-10secs', device=None):
		"""Return pretrained model, we provide six models:
		- facebook/magnet-small-10secs (300M), text to music, 10-second audio samples.
			# see: https://huggingface.co/facebook/magnet-small-10secs
		- facebook/magnet-medium-10secs (1.5B), text to music, 10-second audio samples.
			# see: https://huggingface.co/facebook/magnet-medium-10secs
		- facebook/magnet-small-30secs (300M), text to music, 30-second audio samples.
			# see: https://huggingface.co/facebook/magnet-small-30secs
		- facebook/magnet-medium-30secs (1.5B), text to music, 30-second audio samples.
			# see: https://huggingface.co/facebook/magnet-medium-30secs
		- facebook/audio-magnet-small (300M), text to sound-effect (10-second samples).
			# see: https://huggingface.co/facebook/audio-magnet-small
		- facebook/audio-magnet-medium (1.5B), text to sound-effect (10-second samples).
			# see: https://huggingface.co/facebook/audio-magnet-medium
		"""
		if device is None:
			if torch.cuda.device_count():
				device = 'cuda'
			else:
				device = 'cpu'

		compression_model = load_compression_model(name, device=device)
		lm = load_lm_model_magnet(name, compression_model_frame_rate=int(compression_model.frame_rate), device=device)

		if 'self_wav' in lm.condition_provider.conditioners:
			lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True

		kwargs = {'name': name, 'compression_model': compression_model, 'lm': lm}
		return LoopGen(**kwargs)

	def set_generation_params(self, use_sampling: bool = True, top_k: int = 0,
								top_p: float = 0.9, temperature: float = 3.0,
								max_cfg_coef: float = 10.0, min_cfg_coef: float = 1.0,
								decoding_steps: tp.List[int] = [20, 10, 10, 10],
								span_arrangement: str = 'nonoverlap',
								rescorer: LMModel = None,
								rescore_weights: torch.Tensor | float = 0.7,
								rescorer_temp: torch.Tensor | float = 1.0,
                   				offset: bool = True):
		"""Set the generation parameters for MAGNeT.

		Args:
			use_sampling (bool, optional): Use sampling if True, else do argmax decoding. Defaults to True.
			top_k (int, optional): top_k used for sampling. Defaults to 0.
			top_p (float, optional): top_p used for sampling, when set to 0 top_k is used. Defaults to 0.9.
			temperature (float, optional): Initial softmax temperature parameter. Defaults to 3.0.
			max_cfg_coef (float, optional): Coefficient used for classifier free guidance. Defaults to 10.0.
			min_cfg_coef (float, optional): End coefficient of classifier free guidance annealing. Defaults to 1.0.
			decoding_steps (list of n_q ints, optional): The number of iterative decoding steps, for each of the n_q RVQ codebooks.
		"""
		if isinstance(rescore_weights, int):
			rescore_weights = float(rescore_weights)

		self.generation_params = {
			'use_sampling': use_sampling,
			'temp': temperature,
			'top_k': top_k,
			'top_p': top_p,
			'max_cfg_coef': max_cfg_coef,
			'min_cfg_coef': min_cfg_coef,
			'decoding_steps': [int(s) for s in decoding_steps],
			'rescorer': rescorer,
			'rescore_weights': rescore_weights,
			'rescorer_temp': rescorer_temp,
			'offset' : offset
		}
  
	def generate(self, text_prompt: tp.List[str], negative_text_prompt: tp.Optional[tp.List[tp.Optional[str]]] = None, progress: bool = False, valid_tokens = None) -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
		"""Generate samples conditioned on text.

        Args:
            descriptions (list of str): A list of strings used as text conditioning.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
		if type(text_prompt) is not list:
			text_prompt = [text_prompt]
        
		text_prompt, prompt_tokens = self._prepare_tokens_and_attributes(text_prompt, None)
  
		if valid_tokens is None:
			valid_tokens = torch.ones(len(text_prompt)) * int(self.duration * self.frame_rate)
   
		assert prompt_tokens is None
		if type(negative_text_prompt) is not list:
			negative_text_prompt = [negative_text_prompt] * len(text_prompt)
  
		tokens = self._generate_tokens(valid_tokens, text_prompt, negative_text_prompt, None, progress)
        
		if valid_tokens[0] == int(self.duration * self.frame_rate):
			return self.generate_audio(tokens, valid_tokens, 1)
		else:
			return self.generate_audio(tokens, valid_tokens, 2)

	def generate_continuation(self, left_hint: torch.Tensor, left_hint_sr: int,	text_prompt: tp.Optional[tp.List[tp.Optional[str]]] = None, negative_text_prompt: tp.Optional[tp.List[tp.Optional[str]]] = None,
								progress: bool = False, valid_tokens = None) \
					-> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
		"""Generate samples conditioned on audio prompts and an optional text description.

		Args:
				prompt (torch.Tensor): A batch of waveforms used for continuation.
						Prompt should be [B, C, T], or [C, T] if only one sample is generated.
				prompt_sample_rate (int): Sampling rate of the given audio waveforms.
				descriptions (list of str, optional): A list of strings used as text conditioning. Defaults to None.
				progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
		"""
		# if prompt.dim() == 2:
		#     prompt = prompt[None]
		# if prompt.dim() != 3:
		#     raise ValueError("prompt should have 3 dimensions: [B, C, T] (C = 1).")
		if type(text_prompt) is not list:
			text_prompt = [text_prompt]
   
		if type(left_hint) is not list:
			left_hint = [left_hint]
   
		audio_prompts = left_hint
		left_hint = []
		for p in audio_prompts:
			left_hint.append(convert_audio(p[None], left_hint_sr, self.sample_rate, self.audio_channels))

		if text_prompt is None:
			text_prompt = [None] * len(left_hint)
		if valid_tokens is None:
			valid_tokens = torch.ones(len(left_hint)) * int(self.duration * self.frame_rate)

		text_prompt, left_tokens = self._prepare_tokens_and_attributes(text_prompt, left_hint)
		assert left_tokens is not None
   
		if type(negative_text_prompt) is not list:
			negative_text_prompt = [negative_text_prompt] * len(text_prompt)

		tokens = self._generate_tokens(valid_tokens, text_prompt, negative_text_prompt, left_tokens, progress)
  
		if valid_tokens[0] == int(self.duration * self.frame_rate):
			return self.generate_audio(tokens, valid_tokens, 1)
		else:
			return self.generate_audio(tokens, valid_tokens, 2)

	def _generate_tokens(self, valid_tokens: torch.Tensor, text_prompt: tp.List[ConditioningAttributes], negative_text_prompt: tp.Optional[tp.List[tp.Optional[str]]],
							left_tokens: tp.Optional[torch.Tensor], progress: bool = False) -> torch.Tensor:
		"""Generate discrete audio tokens given audio prompt and/or conditions.

		Args:
				attributes (list of ConditioningAttributes): Conditions used for generation (here text).
				prompt_tokens (torch.Tensor, optional): Audio prompt used for continuation.
				progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
		Returns:
				torch.Tensor: Generated audio, of shape [B, C, T], T is defined by the generation params.
		"""
		total_gen_len = int(self.duration * self.frame_rate)
		max_prompt_len = int(min(self.duration, self.max_duration) * self.frame_rate)
		current_gen_offset: int = 0

		def _progress_callback(generated_tokens: int, tokens_to_generate: int):
			generated_tokens += current_gen_offset
			if self._progress_callback is not None:
				# Note that total_gen_len might be quite wrong depending on the
				# codebook pattern used, but with delay it is almost accurate.
				self._progress_callback(generated_tokens, tokens_to_generate)
			else:
				print(f'{generated_tokens: 6d} / {tokens_to_generate: 6d}', end='\r')
    
		tot_lr_tokens = 0				
		if left_tokens is not None:
			tot_lr_tokens +=left_tokens[0].shape[-1]
		assert max_prompt_len >= tot_lr_tokens, \
		"Audio prompt is longer than available space"


		callback = None
		if progress:
			callback = _progress_callback

		if self.duration <= self.max_duration:
			# generate by sampling from LM, simple case.
			with self.autocast:
				gen_tokens = self.lm.generate(
								valid_tokens, left_tokens, text_prompt, negative_text_prompt,
								callback=callback, max_gen_len=total_gen_len, **self.generation_params)

		else:
			assert False, "Prompt too long, we don't do that here"
		return gen_tokens

	def generate_audio(self, gen_tokens: torch.Tensor, valid_tokens: torch.Tensor, k_loops: int) -> torch.Tensor:
		"""Generate Audio from tokens."""
		assert gen_tokens.dim() == 3
		with torch.no_grad():
			B, K, T = gen_tokens.shape
   
			offset = self.generation_params["offset"]

			left_pads = torch.zeros(B, dtype=torch.long)
			right_pads = torch.zeros(B, dtype=torch.long)
			for i, valid_amount in enumerate(valid_tokens):
				empty_amount = T - valid_amount
				left_pads[i] = min(valid_amount, empty_amount // 2) if offset else 0
				right_pads[i] = empty_amount - left_pads[i]
    
			gc.collect()
			torch.cuda.empty_cache()
			
			gen_audio = self.compression_model.decode(gen_tokens, None)

			gen_audios = []
			for i, (r, l) in enumerate(zip(right_pads, left_pads)):
				if r==0:
					gen_audios.append(torch.cat([gen_audio[i,..., 640 * l:]] * k_loops, -1))
				else:
					gen_audios.append(torch.cat([gen_audio[i,..., 640 * l: -640*r]] * k_loops, -1))

		return gen_audios