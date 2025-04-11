# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import typing as tp
import torch
import tqdm

from copy import deepcopy
from audiocraft.utils import utils
from audiocraft.modules.conditioners import (
    ConditioningAttributes,
    ConditionType,
)
from audiocraft.models.lm import LMModel
from audiocraft.models.lm_magnet import MagnetLMModel

logger = logging.getLogger(__name__)
ConditionTensors = tp.Dict[str, ConditionType]
CFGConditions = tp.Union[ConditionTensors, tp.Tuple[ConditionTensors, ConditionTensors]]


class LoopgenLMModel(MagnetLMModel):

    def __init__(self, subcodes_context: int = 5, compression_model_framerate: int = 50,
                 segment_duration: int = 10, span_len: int = 3, **kwargs):
        super().__init__(subcodes_context, compression_model_framerate,
                 segment_duration, span_len, **kwargs)

    @torch.no_grad()
    def generate(self,
                 valid_tokens: torch.Tensor,
                 left_tokens: tp.Optional[torch.Tensor] = None,
                 conditions: tp.List[ConditioningAttributes] = [],
                 negative_text_prompt: tp.Optional[tp.List[tp.Optional[str]]] = None,
                 num_samples: tp.Optional[int] = None,
                 max_gen_len: int = 256,
                 use_sampling: bool = True,
                 temp: float = 1.0,
                 top_k: int = 250,
                 top_p: float = 0.0,
                 cfg_coef: tp.Optional[float] = None,
                 cfg_coef_beta: tp.Optional[float] = None,
                 two_step_cfg: tp.Optional[bool] = None,
                 remove_prompts: bool = False,
                 check: bool = False,
                 callback: tp.Optional[tp.Callable[[int, int], None]] = None,
                 **kwargs) -> torch.Tensor:

        assert cfg_coef is None, "Unsupported in MAGNeT. Use max_cfg_coef,min_cfg_coef instead."
        assert two_step_cfg is None, "MAGNeT currently doesn't support two step classifier-free-guidance."
        assert remove_prompts is False, "MAGNeT currently doesn't support the remove_prompts arg."
        assert check is False, "MAGNeT currently doesn't support the check arg."
        assert cfg_coef_beta is None, "MAGNeT currently doesn't support the cfg_coef_beta arg."
        # Call the MAGNeT-specific generation method
        return self._generate_magnet(
                                     valid_tokens=valid_tokens,
                                     left_tokens=left_tokens,
                                     conditions=conditions,
                                     negative_text_prompt=negative_text_prompt,
                                     num_samples=num_samples,
                                     max_gen_len=max_gen_len,
                                     use_sampling=use_sampling,
                                     temp=temp,
                                     top_k=top_k,
                                     top_p=top_p,
                                     callback=callback, **kwargs)

    @torch.no_grad()
    def _generate_magnet(self,
                         valid_tokens: torch.Tensor,
                         left_tokens: tp.Optional[torch.Tensor] = None,
                         conditions: tp.List[ConditioningAttributes] = [],
                         negative_text_prompt: tp.Optional[tp.List[tp.Optional[str]]] = None,
                         num_samples: tp.Optional[int] = None,
                         max_gen_len: int = 256,
                         use_sampling: bool = True,
                         temp: float = 3.0,
                         top_k: int = 0,
                         top_p: float = 0.9,
                         callback: tp.Optional[tp.Callable[[int, int], None]] = None,
                         max_cfg_coef: float = 10.0,
                         min_cfg_coef: float = 1.0,
                         decoding_steps: tp.List[int] = [20, 10, 10, 10],
                         anneal_temp: bool = True,
                         rescorer: LMModel = None,
                         rescore_weights: torch.Tensor | float = 0.7,
                         rescorer_temp: torch.Tensor | float = 1.0,
                         offset: bool = True
                         ) -> torch.Tensor:
        """Generate audio tokens given textual conditions, and optionally given audio prompts,
        by running MAGNeT's iterative decoding algorithm for each of the n_q RVQ levels.
        Args:
            prompt (torch.Tensor): Prompt tokens of shape [B, K, T].
            conditions (list of ConditioningAttributes): List of conditions.
            num_samples (int): Number of samples to generate when no prompt and no conditions are given.
            max_gen_len (int): Maximum generation length.
            use_sampling (bool): Whether to use a sampling strategy or not.
            temp (float): Initial sampling temperature.
            top_k (int): k for "top-k" sampling.
            top_p (float): p for "top-p" sampling.
            callback (Callback): Callback function to report generation progress.
            max_clsfg_coef (float): Initial coefficient used for classifier free guidance.
            min_clsfg_coef (float): Final coefficient used for classifier free guidance.
            decoding_steps (list of n_q ints): The number of iterative decoding steps,
                                            for each of the n_q RVQ codebooks.
            anneal_temp (bool): When set to True, softmax temperature will be linearly decayed to zero, at each stage.
        Returns:
            torch.Tensor: Generated tokens.
        """
        assert not self.training, "generation shouldn't be used in training mode."
        first_param = next(iter(self.parameters()))
        device = first_param.device

        # Checking all input shapes are consistent.
        possible_num_samples = []
        if num_samples is not None:
            possible_num_samples.append(num_samples)
        elif left_tokens is not None:
            possible_num_samples.append(len(left_tokens))
        elif conditions:
            possible_num_samples.append(len(conditions))
        else:
            possible_num_samples.append(1)
        assert [x == possible_num_samples[0] for x in possible_num_samples], "Inconsistent inputs shapes"
        num_samples = possible_num_samples[0]

        B, K = num_samples, self.num_codebooks

        mask_id = self.special_token_id

        # we generate codes with a fixed sequence length
        shape = (B, K, max_gen_len)

        gen_codes = torch.full(shape, mask_id, dtype=torch.long, device=device)

        # filling the gen_codes with the prompt if needed
        left_pads = torch.zeros(B, dtype=torch.long)
        right_pads = torch.zeros(B, dtype=torch.long)
        for i, valid_amount in enumerate(valid_tokens):
            empty_amount = max_gen_len - valid_amount
            left_pads[i] = min(valid_amount, empty_amount // 2) if offset else 0
            right_pads[i] = empty_amount - left_pads[i]
        
        rescorer_conditions = None
        if rescorer is not None:
            assert rescorer.special_token_id == mask_id, "Rescorer and generator should have the same mask id."
            rescorer_conditions = rescorer.cfg_dropout(conditions) #* (loop_trick_rotations + 1))
            rescorer_conditions = rescorer.att_dropout(rescorer_conditions)
            rescorer_conditions = rescorer.condition_provider.tokenize(rescorer_conditions)
            # encode conditions and fuse, both have a streaming cache to not recompute when generating.
            rescorer_conditions = rescorer.condition_provider(rescorer_conditions)
                
        if left_tokens is not None:
            for i, tokens in enumerate(left_tokens):
                gen_codes[i,..., left_pads[i]:left_pads[i]+tokens.shape[-1]] = tokens


        # create the gen_sequence with proper interleaving from the pattern: [B, K, S]
        gen_sequence = gen_codes
        
        
        # below we create set of conditions: one conditional and one unconditional
        # to do that we merge the regular condition together with the null condition
        # we then do 1 forward pass instead of 2.
        cfg_conditions: tp.Optional[ConditionTensors]
        if conditions:
            null_conditions = deepcopy(conditions)
            for sample, negative_prompt in zip(null_conditions, negative_text_prompt):
                for condition in sample.attributes["text"]:
                    sample.text[condition] = negative_prompt
            
            tokenized = self.condition_provider.tokenize(conditions + null_conditions)
            cfg_conditions = self.condition_provider(tokenized)
        else:
            cfg_conditions = {}

        curr_step = 0
        pbar = tqdm.tqdm(total=sum(decoding_steps), desc="Generating", leave=False)
        for stage, n_steps in zip(range(self.n_q), decoding_steps):
            gen_sequence, curr_step = self._generate_stage(gen_sequence,
                                                           cfg_conditions,
                                                           stage=stage,
                                                           device=device,
                                                           valid_tokens=valid_tokens,
                                                           left_tokens=left_tokens,
                                                           temp=temp,
                                                           max_cfg_coef=max_cfg_coef,
                                                           min_cfg_coef=min_cfg_coef,
                                                           top_k=top_k,
                                                           top_p=top_p,
                                                           timesteps=n_steps,
                                                           anneal_temp=anneal_temp,
                                                           use_sampling=use_sampling,
                                                           curr_step=curr_step,
                                                           total_steps=sum(decoding_steps),
                                                           callback=callback,
                                                           rescorer=rescorer,
                                                           rescore_weights=rescore_weights,
                                                           rescorer_temp=rescorer_temp,
                                                           rescorer_conditions=rescorer_conditions,
                                                           offset=offset,
                                                           pbar=pbar)
            
        return gen_sequence

    @torch.no_grad()
    def _generate_stage(self,
                        gen_sequence: torch.Tensor,
                        condition_tensors: tp.Optional[ConditionTensors],
                        stage: int,
                        device: torch.device,
                        valid_tokens: torch.Tensor,
                        left_tokens: tp.Optional[torch.Tensor] = None,
                        use_sampling: bool = True,
                        temp: float = 3.0,
                        max_cfg_coef: float = 10.0,
                        min_cfg_coef: float = 1.0,
                        top_k: int = 0,
                        top_p: float = 0.0,
                        timesteps: int = 10,
                        anneal_temp: bool = True,
                        curr_step: int = 0,
                        total_steps: int = 0,
                        callback: tp.Optional[tp.Callable[[int, int], None]] = None,
                        rescorer: LMModel = None,
                        rescore_weights: torch.Tensor | float = 0.7,
                        rescorer_temp: torch.Tensor | float = 1.0,
                        rescorer_conditions = tp.Optional[ConditionTensors],
                        offset: bool = True,
                        pbar: tqdm.tqdm = None) -> tp.Tuple[torch.Tensor, int]:
        """Generate audio tokens of a single RVQ level (stage), given the previously generated stages,
           and the textual conditions.
        Args:
            gen_sequence (torch.Tensor): Previously generated tokens.
            condition_tensors (tp.Optional[ConditionTensors]): pre-computed conditioning tensors.
            stage (int): RVQ level to generate.
            device (torch.device): device of the output tensor.
            prompt_length (int): Temporal length of the audio prompt.
            prompt (torch.Tensor): Prompt tokens of shape [B, K, T].
            use_sampling (bool): Whether to use a sampling strategy or not.
            temp (float): Initial sampling temperature.
            max_clsfg_coef (float): Initial coefficient used for classifier free guidance.
            min_clsfg_coef (float): Final coefficient used for classifier free guidance.
            top_k (int): k for "top-k" sampling.
            top_p (float): p for "top-p" sampling.
            timesteps (int): Number of iterative decoding steps.
            anneal_temp (bool): When set to True, softmax temperature will be linearly decayed to zero, at each stage.
            curr_step (int): Global iterative decoding step counter.
            total_steps (int): Total decoding steps.
            callback (Callback): Callback function to report generation progress.
        Returns:
            tuple(torch.Tensor, int): Generated tokens and the current decoding step counter.
        """
        B, K, T = gen_sequence.shape
        shape = (B, 1, T)  # generating a single codebook per stage

        mask_id = self.special_token_id
        stage_gen_seq = torch.full(shape, mask_id, dtype=torch.long, device=device)

        DONT_REMASK_ME_SCORE = -1e4

        model = self if self._fsdp is None else self._fsdp

        # token-wise scores
        scores = torch.zeros(shape, dtype=torch.float32, device=device)
        # scores[..., :prompt_length] = DONT_REMASK_ME_SCORE
        # gen_T = T - prompt_length
        gen_T = torch.ones(B) * T
        
        left_pads = torch.zeros(B, dtype=torch.long)
        right_pads = torch.zeros(B, dtype=torch.long)
        for i, valid_amount in enumerate(valid_tokens):
            empty_amount = scores.shape[-1] - valid_amount
            left_pads[i] = min(valid_amount, empty_amount // 2) if offset else 0
            right_pads[i] = empty_amount - left_pads[i]
            scores[i, ..., :left_pads[i]] = DONT_REMASK_ME_SCORE
            if right_pads[i] > 0:
                scores[i, ..., -right_pads[i]:] = DONT_REMASK_ME_SCORE
            gen_T[i] -= empty_amount        

        if left_tokens is not None:
            for i, tokens in enumerate(left_tokens):
                scores[i,..., left_pads[i]:left_pads[i]+tokens.shape[-1]] = DONT_REMASK_ME_SCORE
                stage_gen_seq[i,..., left_pads[i]:left_pads[i]+tokens.shape[-1]] = tokens[[stage]]
                gen_T[i] -= tokens.shape[-1]
                
        if isinstance(rescore_weights, float):
            rescore_weights = torch.ones(timesteps, device=device) * rescore_weights
        
        if isinstance(rescorer_temp, float):
            rescorer_temp = torch.ones(timesteps, device=device) * rescorer_temp

        # run MAGNeT iterative decoding for "timesteps" iterations
        for timestep, steps_left in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
            mask_p = torch.cos(timestep * math.pi * 0.5)

            # masking of the k least probable overlapping (stride 1) spans
            mask = torch.concat((
                [self._least_probable_span_masking(scores[[i], :, :], max(int((mask_p * gen_T[i]).item()), 1)).to(device)
                    for i in range(B)]), dim=0)
            stage_gen_seq[mask] = mask_id

            for i, (r, l) in enumerate(zip(right_pads, left_pads)):
                if r > 0:
                    stage_gen_seq[i,..., :l] = stage_gen_seq[i,..., -l-r:-r]
                    stage_gen_seq[i,..., -r:] = torch.cat([stage_gen_seq[i,..., l:-r]] * (r // (T - r - l) + 1), -1)[..., :r]

            gen_sequence[:, [stage], :] = stage_gen_seq
            sequence = gen_sequence          
            
            if condition_tensors:
                sequence = torch.cat([sequence, sequence], dim=0)

            all_logits = model(sequence, [], condition_tensors, stage=stage)

            if condition_tensors:
                # classifier free guidance with annealing
                cond_logits, uncond_logits = all_logits.split(all_logits.shape[0]//2, dim=0)  # [B, K, T, card]
                clsfg_coef = float(mask_p) * max_cfg_coef + (1 - float(mask_p)) * min_cfg_coef
                logits = uncond_logits + (cond_logits - uncond_logits) * clsfg_coef                        
            else:
                logits = all_logits

            # temperature annealing - linear
            t = temp * (steps_left / timesteps) if anneal_temp else temp                

            # sampling
            logits = logits[:, stage, :, :].unsqueeze(1)                
            probs = torch.softmax(logits / max(t, 1e-2), dim=-1)
                
            if use_sampling:
                if top_p > 0.0:
                    sampled_tokens = utils.sample_top_p(probs, p=top_p)
                elif top_k > 0:
                    sampled_tokens = utils.sample_top_k(probs, k=top_k)
                else:
                    sampled_tokens = utils.multinomial(probs, num_samples=1)
            else:
                sampled_tokens = torch.argmax(logits, dim=-1, keepdim=True)

            # place mask_id token in each of the masked positions
            mask = stage_gen_seq == mask_id
            stage_gen_seq = torch.where(mask, sampled_tokens[..., 0], stage_gen_seq)
            gen_sequence[:, [stage], :] = stage_gen_seq

            # get probs of sampled tokens
            sampled_probs = torch.gather(probs, 3, sampled_tokens)[..., 0]
            
            if rescorer:
                rescorer_logits = rescorer.compute_predictions(gen_sequence, conditions=None, condition_tensors=rescorer_conditions).logits[:, [stage]]

                rescorer_probs = torch.softmax(rescorer_logits / rescorer_temp[steps_left], dim=-1)
                rescorer_sampled_probs = torch.gather(rescorer_probs, 3, sampled_tokens)[..., 0]

                sampled_probs = rescore_weights[[steps_left]] * rescorer_sampled_probs + (1 - rescore_weights[[steps_left]]) * sampled_probs

            # prod in log space for lps masking (stride1)
            scores = -torch.log(sampled_probs)

            # Fix unmasked tokens by placing inf probs (-inf scores)
            scores = scores.masked_fill(~mask, DONT_REMASK_ME_SCORE)

            if callback is not None:
                curr_step += 1
                callback(curr_step, total_steps)
            
            if pbar is not None:
                pbar.update(1)

        return gen_sequence, curr_step