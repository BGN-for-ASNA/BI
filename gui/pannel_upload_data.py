#%%
import flet as ft
import pandas as pd
from gui.utils import *
from code import data_manip

#from bayesian.distribution import *
# Upload page ----------------------------------
class upload(ft.UserControl):
    # Link page attributes
    def __init__(self, page: ft.Page):
        super().__init__()
        self.page = page
        self.separator = ';'
             
    def build(self): 
        self.page.on_resize = self.page_resize_up 

        #Text Field to type our task in-------------------------------------------------------------------------------------------------- 
        self.pick = ft.ElevatedButton("Pick files", icon=ft.icons.UPLOAD_FILE,
                   on_click=lambda _: self.pick_files_dialog.pick_files(
                       allow_multiple=False
                   ))
        #self.table =  ft.Column(controls=[], width=self.page.window_height*0.80,scroll=True)   
        #self.tablerow =  ft.Row([self.table],scroll=True,expand=1,vertical_alignment=ft.CrossAxisAlignment.START) 
        self.output = ft.Column(controls=[], width=self.page.window_height*0.80,scroll=True)        
        self.pick_files_dialog = ft.FilePicker(on_result=self.pick_files_result)
        self.selected_files = ft.Text()
        self.page.overlay.append(self.pick_files_dialog)   
        
        self.tableMRD =  ft.Markdown(
                            value = "test",
                            selectable=True,
                            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                            on_tap_link=lambda e: page.launch_url(e.data),
                        )


        # Handling data-------------------------------------------------------------------------------------------------- 
        self.sep = ft.TextField(label="separator", multiline=False, value = ";")  
        
        self.scaleData = ft.Checkbox(label="Center data", on_change=self.checkbox_scale)
        self.scaleCol = ft.TextField(label="Which column(s) to scale", value = "all",  multiline=False) 
        
        self.ohe = ft.Checkbox(label="Convert categorical variables", on_change=self.checkbox_ohe)
        self.catCol = ft.TextField(label="Which column(s) to convert", value = "all",  multiline=False) 
        
        self.removeData = ft.Checkbox(label="Remove NA", on_change=self.checkbox_removeData)
        self.inputData = ft.Checkbox(label="Infers NA", on_change=self.checkbox_inputData)
         

        # Output-------------------------------------------------------------------------------------------------- 
        self.result = ft.ResponsiveRow([ft.Column(col={"sm": 4}, controls=[self.sep,self.pick, 
                                                                           self.scaleData, self.scaleCol,
                                                                           self.ohe, self.catCol, 
                                                                           self.removeData, self.inputData]),
                                       ft.Column(col={"sm": 8}, controls=[self.tableMRD])]) 

        return self.result
    
    # Functions-------------------------
          
    def page_resize_up(self, e):
        self.output.width = self.page.window_height*0.80
        self.update()
        
    def checkbox_scale(self, e):
        value = self.scaleData.value 
        print(value)
        self.update()
            
    def checkbox_ohe(self, e):
        value = self.ohe.value
        if value:
            if self.catCol.value == 'all':
                self.page.df = data_manip.OHE(self.page.df)
                self.tableMRD.value =  self.page.df.iloc[:10].to_markdown(index = False)
            else: 
                result = textInput_to_list(self.catCol.value)          
                self.page.df = data_manip.OHE(self.page.df, cols = result)
                self.tableMRD.value =  self.page.df.iloc[:10].to_markdown(index = False)
        self.update()
        
    def checkbox_removeData(self, e):
        value = self.removeData.value 
        if value:
            self.page.df = self.page.df.dropna()
            self.output.controls = [ft.DataTable(
                columns=headers(self.page.df.iloc[:10]),
                rows=rows(self.page.df.iloc[:10]), width= self.page.window_height*0.80, show_bottom_border=True)]  
            self.output.width = self.page.window_height*0.80

        self.update()     
    
    def checkbox_inputData(self,e):
        self.update()  
    # Pick files dialog
    def pick_files_result(self, e):
        self.selected_files.value = (
            ", ".join(map(lambda f: f.path, e.files)) if e.files else "Cancelled!"
        )
        
        if  self.selected_files.value != "Cancelled!":
            self.page.df = pd.read_csv(str(self.selected_files.value), sep = self.sep.value) 
            self.tableMRD.value =  self.page.df.iloc[:10].to_markdown(index = False)
            self.output.controls = [ft.DataTable(
                columns=headers(self.page.df.iloc[:10]),
                rows=rows(self.page.df.iloc[:10]), width= self.page.window_width*0.80, show_bottom_border=True)]             
            
            self.update()
        else:
            self.selected_files.update()
