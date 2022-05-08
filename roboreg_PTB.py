import pybullet as p
import time
import math
from datetime import datetime
import pybullet_data


p.connect(p.GUI)
  #p.connect(p.SHARED_MEMORY_GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

robot = p.loadURDF('./SpiderBot_URDFs/SpiderBot_4Legs/urdf/SpiderBot_4Legs.urdf', [0, 0, 0.5])
ground = p.loadURDF('plane.urdf')
p.setGravity(0, 0, -9.8)

def moveLeg( robot=None, id=0, position=0, force=1.5  ):
    if(robot is None):
        return
    p.setJointMotorControl2(
        robot,
        id,
        p.POSITION_CONTROL,
        targetPosition=position ,
        force=force,
        #maxVelocity=5
    )


toggle = -1
p.setRealTimeSimulation(1)
