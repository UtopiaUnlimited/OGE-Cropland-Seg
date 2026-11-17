# Lovasz hinge

`cropland/losses.py:lovasz_hinge` adapts the binary Lovasz hinge and Jaccard
gradient algorithm from Maxim Berman's LovaszSoftmax (MIT, copyright 2018).
It uses current PyTorch autograd, explicit float32 arithmetic and mini-batch
pooling. No external package or download is needed at training or inference.

Source: https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py

Paper: Berman, Rannen Triki and Blaschko, *The Lovasz-Softmax loss: A tractable
surrogate for the optimization of the intersection-over-union measure in
neural networks*, CVPR 2018. https://arxiv.org/abs/1705.08790

License: [LICENSE](LICENSE).
