## # Running on SSP Cloud

1. Create a GPU instance such as `Vscode-pytorch-gpu` and launch with default parameters.

2. Open it.

3. Move `setup_env.sh` into the working directory.

4. Open a terminal.

5. Run the command:

   ```sh
   chmod u+x setup_env.sh && ./setup_env.sh
   ```

6. Wait until everything is done (~ 5 minutes). You can now run `run.sh` or create a notebook and start experimenting.

---

## Modifications to the main

- Added possibility to use `llava-onevision-qwen2-0.5b-ov` instead of `llava-onevision-qwen2-7b-ov`.
- Added `custom_builder.py` to allow it to run locally on 5.2 GB of vRAM.
- Added an executable `run.sh` allowing easy model swapping.

---

Running the llava models, see `run.sh` :

```sh
python3 -m src.run \
    --model_name llava_ov_0.5 \
    --data_name natural_ret \
    --train_path data/naturalbench_ret_train.jsonl \
    --val_path data/naturalbench_ret_test.jsonl
```

Switching `llava_ov_0.5` for `llava_ov_7b` will ignore all modifications and use the base code to run `llava-onevision-qwen2-7b-ov` as intended by the author.

---

Other Notes:
 - NaturalBench Images can be downloaded at the following [link](https://huggingface.co/datasets/BaiqiL/naturalbench_pictures/blob/main/raw_images.zip)

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
