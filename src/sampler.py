from typing import List, Optional

import torch
import transformers
import numpy as np
from .utils import _apply_top_k_top_p, _apply_min_p, _multinomial

class RecursiveSampler:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def recursive_sampler_step(self, outputs, beam_len=5, temp=0.9, top_k=50, top_p=0.9, min_p=0.0, do_top_k_top_p=True, do_min_p=True,
                               used_tokens: Optional[List[int]] = None,
                               terminators: Optional[List[int]] = None,
                               generator: torch.Generator = None):
        sum_score = 0
        terminate = -1
        
        for i in range(beam_len):
            outputs = self.model.generate(
                outputs['sequences'],
                temperature=None,
                top_p=None,
                past_key_values=outputs.get("past_key_values"),
                max_new_tokens=1,
                pad_token_id=self.tokenizer.eos_token_id,
                # eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True, output_scores=True,
                do_sample=False
            )
    
            logits = outputs['scores'][0]
            logits.div_(temp)
            probs = torch.softmax(logits, dim=-1, dtype=torch.float)

            if used_tokens:
                if i == 0:
                    probs[:, used_tokens] = -float('inf')
    
            if do_top_k_top_p:
                probs = _apply_top_k_top_p(probs, top_k, top_p)
    
            if do_min_p:
                probs = _apply_min_p(probs, min_p)
    
            # sample_results = _multinomial(probs)
            # print(probs[probs > 0])
            probs[probs < 0] = 0
            sample_results = torch.multinomial(probs, num_samples=1, generator=generator)
            # print(sample_results)
            score = logits[:, sample_results[0][0]].item()
            sum_score += score
    
            # Replace last element (that were just greedy searched)
            outputs['sequences'][:, -1] = sample_results
            
            # Check if the token is a terminator.
            if terminators is not None and sample_results[0][0] in terminators:
                terminate = i
                break
            
        return outputs, sum_score, terminate

    def generate(self, text, max_tokens=256, n_beams=2, beam_len=5, temperature=0.8,
                 top_k=5, top_p=1.0, do_min_p=True, min_p=0.1,
                 streamer: Optional[transformers.TextStreamer] = None,
                ):
        # print(f'Generating recursive completion for text: {text}')
        # Tokenize the input text.
        inputs = self.tokenizer.encode(
                text,
                return_tensors='pt'
        ).to(self.model.device)
    
        net_inputs = {'sequences': inputs}
        scores = np.zeros(n_beams)
        symbols_cnt = np.zeros(n_beams)
        terminates = np.zeros(n_beams)
        
        do_top_k_top_p = True
        top_k = torch.tensor([top_k]).to(self.model.device)
        top_p = torch.tensor([top_p]).to(self.model.device)
        min_p = torch.tensor([min_p]).to(self.model.device)
    
        g = torch.Generator('cuda')
        
        for _ in range(max_tokens // beam_len + 1):
            beams = {}
            used_tokens = []
            for i in range(n_beams):
                test_out_1, score1, terminate1 = self.recursive_sampler_step(net_inputs, beam_len=beam_len,
                                                                        temp=temperature, top_k=top_k, top_p=top_p, min_p=min_p,
                                                                        do_top_k_top_p=do_top_k_top_p, do_min_p=do_min_p,
                                                                        used_tokens=used_tokens,
                                                                        terminators=[self.tokenizer.eos_token_id],
                                                                        generator=g)
                used_tokens.append(test_out_1['sequences'][0, -beam_len].item())
                beams[i] = test_out_1
                scores[i] = score1
                terminates[i] = terminate1
    
                # print(f'[{i}] Score: {score1} - ', tokenizer.decode(test_out_1['sequences'][0, -beam_len:], skip_special_tokens=True, clean_up_tokenization_space=True))
    
            if np.any(terminates != -1):
                symbols_cnt[terminates != -1] += 1
                symbols_cnt[terminates == -1] = beam_len
                # Normalize scores
                scores = scores / symbols_cnt
                max_score = np.argmax(scores)
    
                if terminates[max_score] != -1:
                    net_inputs = beams[max_score]
                    break
                else:
                    better_out = beams[max_score]
            else:
                max_score = np.argmax(scores)
                better_out = beams[max_score]

                if streamer:
                    streamer.put(better_out['sequences'][:, -beam_len:])
                
            # Update net_inputs to the better output
            net_inputs = better_out
        
        # Cut off the first tokens (the input text).
        net_inputs['sequences'] = net_inputs['sequences'][:, inputs.shape[-1]:]
        # Decode the generated text.
        completion_text = self.tokenizer.decode(net_inputs['sequences'][0], skip_special_tokens=True, clean_up_tokenization_space=True)
        # print(f'Generated completion: {completion_text}')

        if streamer:
            streamer.end()
        
        return completion_text