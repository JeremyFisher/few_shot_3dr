# few_shot_3dr
Implementation of "Few-Shot Single-View 3-D Object Reconstruction with Compositional Priors" ECCV'20 paper. 


----------------------------------------------------------------------
General notes
----------------------------------------------------------------------
**The code is largely based on Matryoshka [1] repository [2] and was modified accordingly.**

The 2D encoder used is based on Matryoshka paper [1], however using any other encoder
should give similar results.

The very simple 3D decoder used is based on TL paper [3], however using any other
3D decoder should give similar (most likely better) results.

Datasets are loaded using DatasetCollector.py and DatasetLoader.py.

Models should be first trained on all base categories (see base folder) and then
finetuned on novel categories (see novel folder).

See also howto.txt (modify paths so that they point to the right dirs).

We have provided an improvement of the MCCE method where conditional batch norm is applied
in both encoder and decoder. If you want to use it in your network, simply replace all your batchnorm
layers with the layer defined in mcce.py. Note that you should finetune only `self.embed` 
during finetuning of novel classes.

-------------------------
References
---------------------------
	
[1] https://arxiv.org/abs/1804.10975

[2] https://bitbucket.org/visinf/projects-2018-matryoshka/src/master/

[3] https://arxiv.org/abs/1603.08637

[4] https://arxiv.org/pdf/2004.06302.pdf
