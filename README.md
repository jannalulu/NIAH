# Needle in the Haystack
Run `pass_key.py "hf model path"` to run a single needle in the haystack test. 
Results will be written into `data/`, a heatmap and a CSV.

The `max_depth` (for now) goes from 0-1, and a `max_depth` of 0 means that the needle is placed at the beginning of the text, and a `max_depth` of 1.0 means that the needle is placed at the end of the text.

## References
This repository borrows code from the [Mamba repo](https://github.com/state-spaces/mamba), [LongMamba](https://github.com/Jellyfish042/LongMamba), and the [yarn repo](https://github.com/jquesnelle/yarn).
