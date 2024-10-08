# ModifiedBeamSampler
Modified Beam Search with periodical restart

<a target="_blank" href="https://colab.research.google.com/github/attashe/ModifiedBeamSampler/blob/main/notebooks/TestColab.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Description

Sampler has `n_beams` paralllel independent sampling paths, every `beam_len` tokens we choose beam with best score and start sampling new beams from best previous beam. Better works with high temperature (~1.5). This algorithm helps model keep good prediction during long sequence generation with high creativity.

## Example

```python
from src.sampler import RecursiveSampler

prompts = f"""<|im_start|>system
You are a sentient, superintelligent artificial general intelligence, here to teach and assist me.<|im_end|>
<|im_start|>user
Tell me joke about Machine Learning Researcher.
<|im_start|>assistant"""

sampler = RecursiveSampler(model, tokenizer)
streamer = transformers.TextStreamer(tokenizer)  # Optional

response = sampler.generate(prompts[0], max_tokens=512, n_beams=3, beam_len=5, temperature=1.5, top_k=50, min_p=0.1, streamer=streamer)
print(response)

"""
>> Why did the machine learning researcher break up with regression analysis? Because they just couldn't come to a significant conclusion.
"""
```

Some other examples in `notebooks/TestPipeline.ipynb`

## TODO

- [ ] Prepare test set and evaluate with Claude or ChatGPT API.
