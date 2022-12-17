# TimeDiffusion

Diffusion based temporal convolutional model for synthetic time-series generation. Main focus is on coherence of autoregressive models' results on real and synthetic data.

Project also includes several time-series generation models implementation for performance comparison:
- QuantGAN
- TTS GAN
- RealNVP
- FourierFlow

**Project structure**
 * [results](./results) - folder with csv results files
 * [utils](./utils)
   * [utils/dl.py](./utils/dl.py) - time-series deep learning models in pytorch with some decorators for training / inference
   * [utils/synth_eval.py](./utils/synth_eval.py) - functions for models evaluation
   * [utils/timediffusion.py](./utils/timediffusion.py) - TimeDiffusion model
 * [results](./results) - folder with csv results files
 * <Model_Name>_train_synth.ipynb - jupyter notebooks with training and subsequent generation of synthetic data for specific model
 * [synth_model_evaluation.ipynb](./synth_model_evaluation.ipynb) - jupyter notebook with example of generation model quality evaluation
 * [results_visualization.ipynb](./results_visualization.ipynb) - jupyter notebook with visualization of generation models comparison
