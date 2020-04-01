
class Application(tk.Frame):
    textWidgetList = ['replay_files', 'pickle_files', 'convert_files']
    onlyButtonList = ['convert_files']

    def create_widgets(self):
        self.labels = []
        self.buttons = []
        self.entrys = []
        self.lambdas = []

        for key in self.textWidgetList:
            #Check if only button
            if key not in self.onlyButtonList:
                # Create Textbox, select directory button, and editable path text box
                self.labels.append(tk.Label(self.master, text=key))
                self.buttons.append(tk.Button(self.master, text='Select', command=lambda:))
                self.entrys.append(tk.Entry(self.master))
                


    def __init__(self, master=None):
        self.master = tk.Tk()
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()