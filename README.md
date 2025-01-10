# Sparse Attention Vectors (SAVs)
---
*SAVs are a lightweight few-shot method for extracting truly multimodal features (from image, text, and interleaved inputs) from generative large multimodal models to enable state-of-the-art performance on any vision-language task with discrete labels (e.g. classification or safety alignment*


<p align="center">
  <img src="images/SAVs_banner.png" alt="SAVs Banner"/>
  <br>
  <em><a href="https://chancharikmitra.github.io/SAVs_website/">Website</a> | <a href="https://arxiv.org/abs/2412.00142">Paper</a></em>  
</p>


### Method Overview
---
<p align="center">
  <img src=images/fig2_v8.png />
</p>

Generative Large Multimodal Models (LMMs) like LLaVA and Qwen-VL excel at a wide variety of vision-language (VL) tasks such as image captioning or visual question answering. Despite strong performance, LMMs are not directly suited for foundational discriminative vision-language tasks (i.e., tasks requiring discrete label predictions) such as image classification and multiple-choice VQA. One key challenge in utilizing LMMs for discriminative tasks is the extraction of useful features from generative models. To overcome this issue, we propose an approach for finding features in the model's latent space to more effectively leverage LMMs for discriminative tasks. Toward this end, we present <b>Sparse Attention Vectors (SAVs)</b> -- a finetuning-free method that leverages sparse attention head activations (fewer than 1% of the heads) in LMMs as strong features for VL tasks. With only few-shot examples, SAVs demonstrate state-of-the-art performance compared to a variety of few-shot and finetuned baselines on a collection of discriminative tasks. Our experiments also imply that SAVs can scale in performance with additional examples and generalize to similar tasks, establishing SAVs as both effective and robust multimodal feature representations.

For more information, please refer to our paper!

### 💻 Setup
---
To get started, first clone our repo and set up the environment:

```bash
git clone https://github.com/chancharikmitra/SAVs.git
cd SAVs

conda create -n savs python=3.10 -y
conda activate savs
pip install -e .
```
#### Running SAVs

### 📝 Citation
---
If you found our work useful, please consider starring and citing. Thank you!
```latex
@article{mitra2024sparse,
  title={Sparse Attention Vectors: Generative Multimodal Model Features Are Discriminative Vision-Language Classifiers},
  author={Mitra, Chancharik and Huang, Brandon and Chai, Tianning and Lin, Zhiqiu and Arbelle, Assaf and Feris, Rogerio and Karlinsky, Leonid and Darrell, Trevor and Ramanan, Deva and Herzig, Roei},
  journal={arXiv preprint arXiv:2412.00142},
  year={2024}
}
```
