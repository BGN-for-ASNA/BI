#%%
import flet as ft
import pandas as pd
from gui.utils import *
from bayesian.distribution import *
from gui.pannel_upload_data import*
from gui.pannel_model_def import*
from gui.pannel_model_run import *

def main(page: ft.page):
    page.title = "Bayesian simulation"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.scroll = ft.ScrollMode.ADAPTIVE
    page.bgcolor ="system"
    page.df = pd.DataFrame({'A' : []})    
    # Pages --------------------------------------
    app = TasksApp(page)
    app.visible = False   
    
    up = upload(page)
    up.visible = False   
    
    run = Run(page)
    up.visible = False 
    # Control pages visibility --------------------------------------
    def toUpload(e):
        app.visible = False
        up.visible = True
        False
        page.update()
        
    def toModel(e):
        up.visible = False
        app.visible = True
        run.visible = False
        page.update()
        
    def toRun(e):
        app.visible = False
        up.visible = False
        run.visible = True
        page.update()
        
    bt1 = ft.FilledButton("Upload data", on_click= toUpload, data=app)
    bt2 = ft.FilledButton("Define model", on_click= toModel, data=app)   
    bt3 = ft.FilledButton("Run model", on_click= toRun, data=app)   
        
    page.add(ft.ResponsiveRow([ft.Row([bt1,bt2,bt3])]))
    page.add(app)
    page.add(up)
#ft.app(target=main)

# %%
