from enum import Enum
import math


class LearnableProperties(Enum):
    style: str = 'style'
    object: str = 'object'


PREFIXES = ['a', 'the', 'my']

object_templates_standard = [
    "Professional high-quality photo of {a} {token}. photorealistic, 4k, HQ",
    "A photo of {a} {token}",
    "A photo of {a} {token}. photorealistic, 4k, HQ",
]

object_templates_edits = [
    "Professional high-quality art of {a} {token}. photorealistic",
    "A painting of {a} {token}",
    "A watercolor painting of {a} {token}",
    "A painting of {a} {token} in the style of monet",
    "Colorful graffiti of {a} {token}. photorealistic, 4k, HQ",
    "A line drawing of {a} {token}",
    "Oil painting of {a} {token}",
    "Professional high-quality art of {a} {token} in the style of a cartoon",
    "A close-up photo of {a} {token}",
]

object_templates = object_templates_standard * math.ceil(
    len(object_templates_edits) / len(object_templates_standard)) + object_templates_edits

style_templates = [
    "a painting in the style of {token}",
    "a painting of a dog in the style of {token}",
    "a painting of a cat in the style of {token}",
    "a painting portrait in the style of {token}",
    "a painting of a vase with flowers in the style of {token}",
    "a painting of a valley in the style of {token}",
    "a painting of a fruit bowl in the style of {token}",
    "A painting of a bicycle in the style of {token}",
    "A painting of a pair of shoes  in the style of {token}",
    "A painting portrait of a musician playing a musical instrument in the style of {token}",
    "A painting of a cup of coffee with steam in the style of {token}",
    "A painting close-up painting of a seashell with delicate textures in the style of {token}",
    "A painting of a vintage camera in the style of {token}",
    "A painting of a bouquet of wildflowers in the style of {token}",
    "A painting table set with fine china and silverware in the style of {token}",
    "A painting of a bookshelf filled with books in the style of {token}",
    "A painting close-up painting of a glass jar filled with marbles in the style of {token}",
    "A painting portrait of a dancer captured in mid-motion in the style of {token}",
    "A painting of a collection of antique keys with intricate designs in the style of {token}",
    "A painting of a pair of sunglasses reflecting a scenic landscape in the style of {token}",
]
