# Reconstruct, Inpaint, Finetune: Dynamic Novel-view Synthesis from Monocular Videos

[*Kaihua Chen<sup>*</sup>*](https://www.linkedin.com/in/kaihuac/), [*Tarasha Khurana<sup>*</sup>*](https://www.cs.cmu.edu/~tkhurana/), [*Deva Ramanan*](https://www.cs.cmu.edu/~deva/)

Coming soon... 🤓

![Teaser animation](assets/teaser.gif)

**Abstract** We explore novel-view synthesis for dynamic scenes from monocular videos. Prior approaches rely on costly test-time optimization of 4D representations or do not preserve scene geometry when trained in a feed-forward manner. Our approach is based on three key insights: (1) covisible pixels (that are visible in both the input and target views) can be rendered by first reconstructing the dynamic 3D scene and rendering the reconstruction from the novel-views and (2) hidden pixels in novel views can be "inpainted" with feed-forward 2D video diffusion models. Notably, our video inpainting diffusion model (CogNVS) can be self-supervised from 2D videos, allowing us to train it on a large corpus of in-the-wild videos. This in turn allows for (3) CogNVS to be applied zero-shot to novel test videos via test-time finetuning. We empirically verify that CogNVS outperforms almost all prior art for novel-view synthesis of dynamic scenes from monocular videos.

[**Paper**](https://cog-nvs.github.io/) | [**Project Page**](https://cog-nvs.github.io/)
