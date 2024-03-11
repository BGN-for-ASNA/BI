#%%
import flet as ft
import pandas as pd
from gui.utils import *
from bayesian.distribution import *
from src.model_write import *
from src.model_fit import *
from src.model_diagnostic import *

# Model definition page ----------------------------------
class Run(ft.UserControl):
    # Link page attributes
    def __init__(self, page: ft.Page):
        super().__init__()
        self.page = page
        self.page.df = page.df  

        
    #Build must be implemented which will return the UserTextField along with
    #The Add Button
    def build(self):
        
        #Text Field to type our task in
        self.textField = ft.TextField(label="Write equation", multiline=True, width=self.page.window_width*0.70)  

        taskRow=ft.ResponsiveRow([
                    ft.Column(col={"sm": 11, "md": 11, "xl": 9},
                     controls=[ft.Row(controls=[self.textField,])])])

        return taskRow

