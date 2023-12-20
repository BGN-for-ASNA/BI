#%%
import flet as ft
from gui.gui import main
from code.model_write import *
from code.model_fit import *
from code.model_diagnostic import *
from code.data_manip import OHE
ft.app(target=main)

#%%
from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))
printmd('**BOLD TEXT**')
equation_str= '$$c = \sqrt{a^2 + b^2}$$'
printmd(f'Equation:\n{equation_str}')
a = '9'
b = '12'
import math
c = math.sqrt(int(a)**2 + int(b)**2)
equation_str = equation_str.replace('a',a).replace('b',b).replace('c',str(c))
printmd(f'Equation using assigned variables:\n{equation_str}')