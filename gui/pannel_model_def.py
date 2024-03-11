#%%
import flet as ft
import pandas as pd
from gui.utils import *
from bayesian.distribution import *
from src.model_write import *
from src.model_fit import *
from src.model_diagnostic import *

# Model definition page ----------------------------------
class TasksApp(ft.UserControl):
    # Link page attributes
    def __init__(self, page: ft.Page):
        super().__init__()
        self.page = page
        self.page.df = page.df  
        self.priorN = 0  
        self.main = 0
        
    #Build must be implemented which will return the UserTextField along with
    #The Add Button
    def build(self):
        self.popup = ft.AlertDialog(title='', open=False)          
        self.page.on_resize = self.page_resize       
        #Column that will contain every created Task
        self.tasks = ft.Column()
        
        #Text Field to type our task in
        self.textField = ft.TextField(label="Write equation", multiline=True, width=self.page.window_width*0.70)  
        
        self.Equation =  ft.Markdown(
                value = '',
                selectable=True,
                extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                on_tap_link=lambda e: page.launch_url(e.data),
            )
        #if self.page.df.empty:
        #    #self.dfCol = ft.TextField(label="data frame columns available", value= 'None', width=self.page.window_width*0.70)  
        #else:
        #    #self.dfCol = ft.TextField(label="data frame columns available", value= ''.join(self.page.df.columns), width=self.page.window_width*0.70)
        #    self.Equation.value =  self.page.df.iloc[:10].to_markdown(index = False)  
        
        self.distributions =  ft.ListView(expand=1, spacing=10, padding=20, auto_scroll=False)
        for i in range(len(distributions)):
            if(i == 1):
                next
            self.distributions.controls.append(
                ft.ElevatedButton("\t" + distributions[i], style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10),)))
               
        self.full_model = {}
        self.validate_model = ft.FilledButton(text="Validate model", on_click=self.validate_model)

        self.distList = ft.Row(controls=[self.distributions], height=self.page.window_height*0.80)
        
        taskRow=ft.ResponsiveRow([
                    ft.Column(col={"sm": 11, "md": 11, "xl": 9},
                     controls=[ft.Row(controls=[self.textField,]),                               
                                self.validate_model, self.popup, self.Equation]),
                    ft.Column(col={"sm": 11, "md": 11, "xl": 3},
                        controls=[ft.Text("Available distributions:"), self.distList])                    
                    ])

        return taskRow
    # Functions-------------------------
    def page_resize(self, e):
        self.distList.height = self.page.window_height*0.80
        self.textField.width = self.page.window_width*0.7
        self.update()
     
    def helpD(self, dist):
        #print(("\n".join(arguments(dist, True))))
        #self.distInfo.title = ("\n".join(arguments(dist, True)))
        #self.distInfo.open = True
        #self.page.add(self.distInfo)
        self.update()  
             
    def validate_model(self, e):  
        self.Equation.value = self.textField.value.splitlines()
        # Initialize an empty dictionary
        result_dict = {}
        input_list = self.textField.value.splitlines()
        
        
        for item in input_list:
            # Split the string into variable name and distribution
            variable, distribution = item.split('=')

            # Remove extra whitespace and quotes
            variable = variable.strip()
            distribution = distribution.strip(" ',")

            # Add the key-value pair to the dictionary
            result_dict[variable] = distribution    

        self.dic = result_dict
        if self.page.df.empty :
            self.model = build_model(self.dic , path = None, sep = ',', float=32)             
        else:
             self.model = build_model(self.dic , path = None, df = self.page.df, sep = ',', float=32)
             
        if self.model == None:
            text = "Model mispecified"
        else:
            text = "Model have been build"
            
        self.popup = ft.AlertDialog(title=ft.Text(text) )
        self.popup.open = True
        self.page.add(self.popup)
 
        self.update()
     
# %%
