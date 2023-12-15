#%%
import flet as ft
import pandas as pd
from gui.utils import *
#from bayesian.distribution import *
# Upload page ----------------------------------
class upload(ft.UserControl):
    # Link page attributes
    def __init__(self, page: ft.Page):
        super().__init__()
        self.page = page
        self.separator = ';'
             
    def build(self):        
        #Text Field to type our task in     
        self.output = ft.Column(controls=[])        
        self.pick_files_dialog = ft.FilePicker(on_result=self.pick_files_result)
        self.selected_files = ft.Text()
        self.page.overlay.append(self.pick_files_dialog)   
        
        # Handling data-------------------------------------------------------------------------------------------------- 
        self.sep = ft.TextField(label="separator", multiline=False, value = ";")  
        self.scaleData = ft.Checkbox(label="Center data", on_change=self.checkbox_changed)
        self.removeData = ft.Checkbox(label="Remove NA", on_change=self.checkbox_changed)
        self.inputData = ft.Checkbox(label="Infers NA", on_change=self.checkbox_changed)
        result = ft.ResponsiveRow([
            ft.Column(col={"sm": 11, "md": 11, "xl": 9}, 
                    controls = [ft.Row([ft.ElevatedButton(
                    "Pick files",
                    icon=ft.icons.UPLOAD_FILE,
                    on_click=lambda _: self.pick_files_dialog.pick_files(
                        allow_multiple=False
                    )),
                self.selected_files,                
            ]), ft.Row([self.output])]), 
            ft.Column(col={"sm": 11, "md": 11, "xl": 3}, controls=[self.sep, self.scaleData, self.removeData, self.inputData])
        ])
        return result
    
    def checkbox_changed(self, e):
        value = f"Checkbox value changed to {self.scaleData.value}" 
        print(value)
        self.update()
    
    # Pick files dialog
    def pick_files_result(self, e):
        self.selected_files.value = (
            ", ".join(map(lambda f: f.path, e.files)) if e.files else "Cancelled!"
        )
        
        if  self.selected_files.value != "Cancelled!":
            self.page.df = pd.read_csv(str(self.selected_files.value), sep = self.sep.value) 
            self.output.controls.append( ft.DataTable(
                columns=headers(self.page.df.iloc[:10]),
                rows=rows(self.page.df.iloc[:10])))            
            self.update()
        else:
            self.selected_files.update()
