# Copyright 2022 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from rich.console import Console
from rich.markdown import Text
from rich.style import Style
from rich.color import Color

from rich.progress import ProgressColumn

import colorsys

console = Console()

def color_good_bad(value:float):
    """
    Returns a rgb color interpolated between 
    green if value=0 and red if value=1
    """
    # hue: 0 -> red, 0.4->green
    h = (1-value) * 0.4
    # luminance
    l = 0.9
    # saturation
    s = 0.9

    r,g,b=colorsys.hsv_to_rgb(h, l, s)
    # hsb to rgb
    return Color.from_rgb(r*255,g*255,b*255) 


def __repr_from_rich__(self):
    """
    default __repr__ that calls __rich__
    """
    import rich
    console = rich.get_console()
    segments = list(console.render(self, console.options))
    text = console._render_buffer(segments)
    return text

def rich_repr(clz):
    """
    Class decorator setting the repr method to use
    `rich`.
    """
    setattr(clz, "__repr__", __repr_from_rich__)
    return clz
