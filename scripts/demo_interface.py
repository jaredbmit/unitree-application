from flask import Flask, render_template, request
import os
import sys
import time
import numpy as np
base_dir = os.path.expanduser("~/drl/unitree-application")
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.unitree import Unitree

app = Flask(__name__)

pos_array = ([[0,1,1],[0,1,1],[0,1,1],[0,1,1],[0,1,1],[0,1,1],[0.600,0.05,0.08],[0.600,-0.05,0.08],[0.600,-0.15,0.08]])

@app.route('/')
def index():
  print("called index")
  return render_template('index.html')


#Stack bricks according to pattern when the pattern is sent...
@app.route('/server_process_layout/', methods=['POST'])  
def server_process_layout():
  #The part that handles request
  data = request.get_json()
  print("data: " + str(data))
  print("running..")
  print("len(data): " + str(len(data)))
  data = data["brick_values"]
  for i in range(len(data) - 1):
      print("i === " + str(i))
      print("brick color: " + str(data[len(data) - 1 -i]))
      if data[len(data) - 1 - i] == 2:
          print("grabbing red brick..")
          unitree.place_color('r',pos_array[i])
      elif data[len(data) - 1 - i] == 1:
          print("grabbing grey brick..")
          unitree.place_color('g', pos_array[i]) 
  return 'trajectory completed successfully'

if __name__ == '__main__':
  #Robot setup
  robot_id = 165
  unitree = Unitree(robot_id)
  unitree.move_home()
  app.run(use_reloader=False, host="0.0.0.0")

unitree.stop()

