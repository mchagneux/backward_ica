# Sequential VAEs 

## TODO 
Re-add the sequential factorization with a very simple dynamical model on the latents
- [x] Consider latents as observations and run Kalman smoothing on them: *Good behaviour expect at some parts*
- [ ] Train a sequential VAE, with:
    - [ ] Frozen image-by-image VAE and we train the GRU (decoder frozen)
    - [ ] Pretrained image-by-image VAE and we finetune GRU + encoder (decoder frozen) 
    - [ ] (Same but decoder unfrozen)
    - [ ] For the classifier part: (i) run it on smoothed latents (ii) retrain it on smoothed latents

Analyze the evolution of the latents with / without smoothing.
- How do they evolve w.r.t the motion of the objects ? 
- Is it continuous / nonlinear, etc. 
- Cosine sim. / tSNE, etc 


