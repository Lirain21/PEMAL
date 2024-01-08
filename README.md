

<h1> PEMAL: Physical formula enhanced multi-task learning for
pharmacokinetics prediction </h1>


Implementation for arXiv:  [GraphMAE: Self-Supervised Masked Graph Autoencoders](https://arxiv.org/abs/2205.10803).


GraphMAE is a generative self-supervised graph learning method, which achieves competitive or better performance than existing contrastive methods on tasks including *node classification*, *graph classification*, and *molecular property prediction*.




<h2>Dependencies </h2>

* Python >= 3.7
* [Pytorch](https://pytorch.org/) >= 1.9.0 
* [dgl](https://www.dgl.ai/) >= 0.7.2
* pyyaml == 5.4.1


<h2>Quick Start </h2>


**Supported datasets**:

* Stage I: Pre-training on unlabeled molecular structures :  `ic50`
* Stage II: Pre-training on labeled but noisy pharmacokinetic data.   `CL `,  `Vdss ` and  `T1_2 `
* Stage III: Physical formula enhanced multi-task learning :  `PK_Mol `

**Stage I**
python chem/pretraining.py

**Stage II**
python chem/finetune_reg.py

**Stage III**
python chem/physical_equation_4_tasks_con.py

**Adjusting hyperparameter**
python single_auto_scripts_reg.py










