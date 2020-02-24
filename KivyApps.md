import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout 
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.config import Config

Config.set('graphics', 'width', '300')
Config.set('graphics', 'height', '300')
Config.set('graphics', 'resizable', '0')

SaveInput=""

import random

import numpy as np
class CalculatorApp(App):
	def build(self):
		root=BoxLayout(orientation='vertical', padding=1)
		
		self.result=TextInput(readonly=True, font_size=24, size_hint=[1, .75], background_color=[1,1,1,.8])
		root.add_widget(self.result)
		

		allButtons=GridLayout(cols=5)
		allButtons_list=['yes','8','9','+','<','4','5','6']
		
		for button in allButtons_list:
			allButtons.add_widget(Button(text=button,on_press=self.calculate))
		root.add_widget(allButtons)

		return root

	def calculate(self, symbol):
		global SaveInput
		if symbol.text=='yes':
			SaveInput=self.result.text=str(np.random.randint(0,5) )#"bitch"
		elif symbol.text!='=':
			self.result.text+=symbol.text
			SaveInput+=symbol.text
		else:
			try:
				SaveInput=self.result.text=str(eval(SaveInput))
			except:
				SaveInput=self.result.text=""

if __name__=='__main__':
	CalculatorApp().run()
