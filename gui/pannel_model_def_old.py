#%%
import flet as ft
import pandas as pd
from gui.utils import *
from bayesian.distribution import *

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
        self.textField = ft.TextField(label="Write equation")
        
        #Define Add Button with an Icon next to it
        #This will trigger addClicked method when clicked
        self.addBtn = ft.FloatingActionButton(icon= ft.icons.ADD, 
                                              on_click=self.addClicked,
                                              tooltip= "Add equation to model")
           
        self.distributions =  ft.ListView(expand=1, spacing=10, padding=20, auto_scroll=False)
        for i in range(len(distributions)):
            if(i == 1):
                next
            self.distributions.controls.append(
                ft.ElevatedButton("\t" + distributions[i], style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=10),)))
               
        self.full_model = {}
        self.validate_model = ft.FilledButton(text="Validate model", on_click=self.validate_model)
        #Create a row containing your TextField and the button you 
        #Created Above
        self.distList = ft.Row(controls=[self.distributions], height=self.page.window_height*0.80)
        
        taskRow=ft.ResponsiveRow([
                    ft.Column(col={"sm": 6, "md": 6, "xl": 9},
                     controls=[ft.Row(controls=[self.textField, self.addBtn, ]),
                        self.tasks,
                        self.validate_model, self.popup]),
                    ft.Column(col={"sm": 2, "md": 2, "xl": 3},
                        controls=[ft.Text("Available distributions:"), self.distList]),])

        return taskRow

    # On page resize change item size
    def page_resize(self, e):
        self.distList.height = self.page.window_height*0.80
        self.update()
     
    def helpD(self, dist):
        #print(("\n".join(arguments(dist, True))))
        #self.distInfo.title = ("\n".join(arguments(dist, True)))
        #self.distInfo.open = True
        #self.page.add(self.distInfo)
        self.update()  
             
    def validate_model(self, e):        
        self.full_model = {}
        
        for a in range(len(self.tasks.controls)):
            if self.tasks.controls[a].eqType == 'Prior':
                self.tasks.controls[a].eqType = self.tasks.controls[a].eqType + str(self.priorN)
                self.priorN += 1
                print(str(self.tasks.controls[a].eqType))
            
            if self.tasks.controls[a].eqType == 'Main':
                self.tasks.controls[a].eqType = self.tasks.controls[a].eqType + str(self.main)
                self.main += 1
                print(str(self.tasks.controls[a].eqType))
                
            self.full_model[self.tasks.controls[a].eqType] = dict(
                 input = self.tasks.controls[a].taskName,
                 model = self.tasks.controls[a].model
            )
            
        
        if self.page.df.empty != True:
            test = undeclared_params(self.full_model, self.page.df)
            text = 'All parameters are correctly defined.'
            print("Var with likelihood")
            print(test)
            if len(test['undeclared_params']) == 0:
                text = 'All parameters are correctly defined.\n You can run the model.' 
                print("Parameters to include in the dataframe: " + ','.join(test['params_in_data']))
                write_model(self.full_model, self.page.df)
            else:            
                if len(test['undeclared_params']) == 1:
                    text = str('The following parameter is not defined: ' + ','.join(test['undeclared_params']))
                if len(test['undeclared_params']) > 1:
                    text = str('The following parameters are not defined: ' + ','.join(test['undeclared_params']))
        else: #No var model  
            if 'likelihood' in self.full_model:
                print("No var with likelihood")
                text = 'All parameters are correctly defined.\n You can run the model.'  
                write_nonVar_withL(self.full_model)
            else:     
                print("No var without likelihood")
                text = 'All parameters are correctly defined.\n You can run the model.' 
                write_nonVar_model(self.full_model)
                
        self.popup = ft.AlertDialog(title=ft.Text(text) )
        self.popup.open = True
        self.page.add(self.popup)
        self.update()
        print(self.full_model)
    
    #Function to be triggered when clicking Add Button
    def addClicked(self,e):
        #Create a task Object (we will program it later below)
        #It receives the text the user typed, and a we pass
        #taskDelete Methode to it. Which will be used as a reference
        #To delete the task
        task=Task(self.textField.value, self.taskDelete)
        self.tasks.controls.append(task)        
        #After typing in a task, reset the textField to be ready for 
        #a new task to be written
        self.textField.value=""
       
        #Call update method from the inherited UserControl to
        #Update the view
        self.update()
        
    #Method to be passed to Task Object which will remove
    #The task from the View when pressing Delete
    def taskDelete(self,task):
        self.tasks.controls.remove(task)
        self.update()    
    
class Task(ft.UserControl): 
    #Call super since we are passing extra parameters to the
    #Inherited UserControl
    #Handle TaskName(The text we typed)
    #And taskDelete (The task delete function reference we passed above)
    def __init__(self, taskName,taskDelete):
        super().__init__()
        self.taskName=taskName
        self.taskDelete= taskDelete
        self.eqType = ''  

    def build(self):
        self.displayTask =ft.TextField(label= self.taskName)
        self.editName = ft.TextField()    
        self.dropdown = ft.Dropdown(width=150, 
                                    options=[ft.dropdown.Option("Main"),
                                             ft.dropdown.Option("likelihood"),
                                             ft.dropdown.Option("Prior"),
                                            ],
                                    autofocus=True,
                                    on_change=self.dropdown_changed)
        
        try:
            dist = get_formula(formula = self.taskName, type = '!likelihood')[1]
            dist = dist.replace(' ', '')
            text = arguments(dist, False)
            text = ', '.join(text)
            self.text = ft.Text( "Arguments order : " + text)
            self.help = ft.ElevatedButton(dist + ' help', on_click= self.help)
            
        except ValueError as e:
            self.text = ft.Text("No distribution declared or no distribution recognized")
            self.help = ft.ElevatedButton('No distribution', disabled=True)
            
        self.displayView = ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN,controls=[self.displayTask, self.dropdown,self.help,
                                            ft.Row(controls=[
                                                    #ft.IconButton(ft.icons.CREATE_OUTLINED,
                                                    #              on_click=self.editClick),,
                                                    ft.IconButton(ft.icons.DELETE_OUTLINED,
                                                                  on_click=self.deleteClick)
                                                  ]
                                            )],spacing = 50)             

        self.editView = ft.Row(visible=False,controls=[self.editName, self.dropdown,self.help,
                                         ft.IconButton(icon=ft.icons.DONE_OUTLINED,
                                                       on_click=self.saveClick)])

        return ft.Column(controls=[self.displayView, self.editView, self.text])

    def help(self, e):
        dist = get_formula(formula = self.taskName, type = '!likelihood')[1]
        dist = dist.replace(' ', '')
        self.help = ft.AlertDialog(
            title=ft.Text(
                ("\n".join(arguments(dist, True)))
            )
        )        
        self.help.open = True
        self.page.add(self.help)
        
        self.update()        
    
    def dropdown_changed(self,e):
        self.eqType = self.dropdown.value
        try:
            self.model = get_formula(self.taskName, self.eqType) 

        except ValueError as e:
            self.error = ft.AlertDialog(
                    title=ft.Text("Equation doesn't look like a " + self.eqType + " equation.\n" +
                                  "Main and priors should be in the following form:\n" ,
                                  size = 22,
                                  spans=[ft.TextSpan(
                                        "\t var ~ distribution(arguments)\n", 
                                        ft.TextStyle(italic=True, size= 22 )),
                                        ft.TextSpan(
                                        "Likelihood should be in the following form:\n", 
                                        ft.TextStyle(italic=True , size= 22 )),  
                                        ft.TextSpan(
                                        '\t y = a + b * c + (1| d)', 
                                        ft.TextStyle(italic=True , size= 22 ))                                        
                                        ])
            )        
            self.error.open = True
            self.page.add(self.error)
            self.update()           
        
    def editClick(self,e):
        self.editName.value=self.displayTask.label
        self.displayView.visible=False
        self.editView.visible=True
        self.update()
        
    def saveClick(self,e):
        self.displayTask.label=self.editName.value
        self.displayView.visible=True
        self.editView.visible=False
        self.update()
    
    def type(self,e):
        self.displayTask.label=self.editName.value
        self.displayView.visible=True
        self.editView.visible=False
        self.update()
    
    def deleteClick(self,e):
        self.taskDelete(self)
            
class Task(ft.UserControl): 
    #Call super since we are passing extra parameters to the
    #Inherited UserControl
    #Handle TaskName(The text we typed)
    #And taskDelete (The task delete function reference we passed above)
    def __init__(self, taskName,taskDelete):
        super().__init__()
        self.taskName=taskName
        self.taskDelete= taskDelete
        self.eqType = ''  

    def build(self):
        self.displayTask =ft.TextField(label= self.taskName)
        self.editName = ft.TextField()    
        self.dropdown = ft.Dropdown(width=150, 
                                    options=[ft.dropdown.Option("Main"),
                                             ft.dropdown.Option("likelihood"),
                                             ft.dropdown.Option("Prior"),
                                            ],
                                    autofocus=True,
                                    on_change=self.dropdown_changed)
        
        try:
            dist = get_formula(formula = self.taskName, type = '!likelihood')[1]
            dist = dist.replace(' ', '')
            text = arguments(dist, False)
            text = ', '.join(text)
            self.text = ft.Text( "Arguments order : " + text)
            self.help = ft.ElevatedButton(dist + ' help', on_click= self.help)
            
        except ValueError as e:
            self.text = ft.Text("No distribution declared or no distribution recognized")
            self.help = ft.ElevatedButton('No distribution', disabled=True)
            
        self.displayView = ft.Row(alignment=ft.MainAxisAlignment.SPACE_BETWEEN,controls=[self.displayTask, self.dropdown,self.help,
                                            ft.Row(controls=[
                                                    #ft.IconButton(ft.icons.CREATE_OUTLINED,
                                                    #              on_click=self.editClick),,
                                                    ft.IconButton(ft.icons.DELETE_OUTLINED,
                                                                  on_click=self.deleteClick)
                                                  ]
                                            )],spacing = 50)             

        self.editView = ft.Row(visible=False,controls=[self.editName, self.dropdown,self.help,
                                         ft.IconButton(icon=ft.icons.DONE_OUTLINED,
                                                       on_click=self.saveClick)])

        return ft.Column(controls=[self.displayView, self.editView, self.text])

    def help(self, e):
        dist = get_formula(formula = self.taskName, type = '!likelihood')[1]
        dist = dist.replace(' ', '')
        self.help = ft.AlertDialog(
            title=ft.Text(
                ("\n".join(arguments(dist, True)))
            )
        )        
        self.help.open = True
        self.page.add(self.help)
        
        self.update()        
    
    def dropdown_changed(self,e):
        self.eqType = self.dropdown.value
        try:
            self.model = get_formula(self.taskName, self.eqType) 

        except ValueError as e:
            self.error = ft.AlertDialog(
                    title=ft.Text("Equation doesn't look like a " + self.eqType + " equation.\n" +
                                  "Main and priors should be in the following form:\n" ,
                                  size = 22,
                                  spans=[ft.TextSpan(
                                        "\t var ~ distribution(arguments)\n", 
                                        ft.TextStyle(italic=True, size= 22 )),
                                        ft.TextSpan(
                                        "Likelihood should be in the following form:\n", 
                                        ft.TextStyle(italic=True , size= 22 )),  
                                        ft.TextSpan(
                                        '\t y = a + b * c + (1| d)', 
                                        ft.TextStyle(italic=True , size= 22 ))                                        
                                        ])
            )        
            self.error.open = True
            self.page.add(self.error)
            self.update()           
        
    def editClick(self,e):
        self.editName.value=self.displayTask.label
        self.displayView.visible=False
        self.editView.visible=True
        self.update()
        
    def saveClick(self,e):
        self.displayTask.label=self.editName.value
        self.displayView.visible=True
        self.editView.visible=False
        self.update()
    
    def type(self,e):
        self.displayTask.label=self.editName.value
        self.displayView.visible=True
        self.editView.visible=False
        self.update()
    
    def deleteClick(self,e):
        self.taskDelete(self)
        