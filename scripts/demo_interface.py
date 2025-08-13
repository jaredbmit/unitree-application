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


@app.route('/')
def index():
  print("called index")
  return render_template('index.html')


#Stack bricks according to pattern when the pattern is sent...
@app.route('/server_process_layout/', methods=['POST'])  
def server_process_layout():
  #The part that handles request
  print ('server side --- great success!')
  data = request.get_json()
  print("data: " + str(data))
  p_place = np.array([0.52, 0., 0.075])
  unitree.pick_and_place(p_place)
  return 'trajectory completed successfully'

if __name__ == '__main__':
  #Robot setup
  robot_id = 165
  unitree = Unitree(robot_id)
  unitree.move_home()
  print("made it to here!")
  app.run(use_reloader=False, host="0.0.0.0")

unitree.stop()

