## 1. Evaluation

We provide evaluation code to reproduce the quantitative results reported in the paper on Kubric-4D, ParallelDomain4D, and DyCheck.

Download the prediction results (including CogNVS, TrajCrafter, GCD, Shape-of-Motion, Mosca, etc.) from
[cognvs_eval_results](https://huggingface.co/datasets/kaihuac/cognvs_eval_results).

Example: evaluate CogNVS on Kubric-4D

```bash
python eval_metrics.py \
    --base_path cognvs_eval_results/ \
    --dataset kubric_4d \
    --pred_method cognvs
```

To evaluate all methods on all datasets, set the correct `BASE_PATH` in
`run_all_eval.sh` and run:

```bash
sh run_all_eval.sh
```

All evaluation results will be written to `eval_log.log`, corresponding to the
quantitative tables reported in the paper.

⚠️ Notes:
1. Numerical differences from the paper will occur due to data processing details, but they are within reasonable tolerance and do not affect conclusions and rankings.
2. For **Kubric4D** and **Pardom4D**, we obtain the ground-truth renders from
   [GCD](https://github.com/basilevh/gcd), using subsets of 20 sequences for each
   dataset (specifically `eval/list/kubric_test20.txt` and
   `eval/list/pardom_test20.txt` in the GCD codebase). These datasets are evaluated
   at a resolution of `384(H) × 576(W)`.

3. For **DyCheck**, we obtain the MegaSAM renders by following the next
   section. DyCheck is evaluated at a resolution of `960(H) × 720(W)`.
   
## 2. MegaSAM render on DyCheck

For `MegaSAM_render` on DyCheck, we stack the background, and optimize the camera at test time (`tto_cam`, a procedure similar to `render_test_tto` in `mosca_evaluate.py` from the [MoSca](https://github.com/JiahuiLei/MoSca) codebase). To reproduce this process, see `dycheck_eval_render.py`. Running this script requires [PyTorch3D](https://pytorch3d.org/); if it is difficult to install under the cognvs environment, we recommend using a separate environment.

Download the data required to create DyCheck renders from [dycheck_render_inputs](https://huggingface.co/datasets/kaihuac/dycheck_render_inputs). 

Example: render the `paper-windmill` sequence:

```bash
python dycheck_eval_render.py \
    --dycheck_4d_base_path ./dycheck_render_inputs \
    --eval_seq_names paper-windmill
```

