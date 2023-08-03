# ConceptLab: Creative Generation using Diffusion Prior Constraints
> Elad Richardson, Kfir Goldberg, Yuval Alaluf, Daniel Cohen-Or  
> Tel Aviv University  
> Recent text-to-image generative models have enabled us to transform our words into vibrant, captivating imagery. The surge of personalization techniques that has followed has also allowed us to imagine unique concepts in new scenes. However, an intriguing question remains: How can we generate a <i>new</i>, imaginary concept that has never been seen before? In this paper, we present the task of <i>creative text-to-image generation</i>, where we seek to generate new members of a broad category  (e.g., generating a pet that differs from all existing pets). We leverage the under-studied Diffusion Prior models and show that the creative generation problem can be formulated as an optimization process over the output space of the diffusion prior, resulting in a set of "prior constraints". To keep our generated concept from converging into existing members, we incorporate a question-answering model that adaptively adds new constraints to the optimization problem, encouraging the model to discover increasingly more unique creations. Finally, we show that our prior constraints can also serve as a strong mixing mechanism allowing us to create hybrids between generated concepts, introducing even more flexibility into the creative process.

<a href=""><img src="https://img.shields.io/badge/arXiv-2301.13826-b31b1b.svg" height=22.5></a>
<a href="https://kfirgoldberg.github.io/ConceptLab/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 

<p align="center">
<img src="docs/teaser.jpg" width="800px"/>  
<br>
New pets generated using ConceptLab. Each pair depicts a learned concept that was optimized to be novel and not match existing members of the pet category. Running our method with different seeds allows us to generate a variety of different brand-new concepts.
</p>

# Code Coming Soon!

## Creative Generation

<p align="center">
<img src="docs/sealrat.jpg" width="800px"/>  
<img src="docs/suncrest_lizard.jpg" width="800px"/>  
<img src="docs/250_214.jpg" width="800px"/>  
<img src="docs/250_211.jpg" width="800px"/>  
<br>
Sample text-guided creative generation results and edits obtained with ConceptLab.
</p>


## Evolutionary Generation
<p align="center">
<img src="docs/mix_tree.jpg" width="800px"/>  
<br>
ConceptLab can be used to mix up generated concepts to iteratively learn new unique creations. This process can be repeated to create further "Generations", each one being a hybrid between the previous two.
</p>


## Concept Mixing
<p align="center">
<img src="docs/concept_mixing.jpg" width="800px"/>  
<br>
With ConceptLab, we can also form hybrid concepts by merging unique traits across multiple real concepts. This can be done by defining multiple positive concepts, allowing us to create unique creations such as a lobs-turtle, pine-melon, and more!
</p>
