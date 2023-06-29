"""control_robot controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())
max_speed = 0
#camera = robot.getDevice("camera")
#camera.enable(timestep)
lidar = robot.getDevice("Hokuyo URG-04LX-UG01")
lidar.enable(timestep)
lidar.enablePointCloud()
left_motor = robot.getDevice('wheel_left_joint')
right_motor = robot.getDevice('wheel_right_joint')
#camera_motor = robot.getDevice("camera_motor")
left_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setPosition(float('inf'))
right_motor.setVelocity(0.0)
#camera_motor.setPosition(float('inf'))
#camera_motor.setVelocity(0.0)
# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    left_motor.setPosition(float('inf'))
    left_motor.setVelocity(max_speed*0.5)
    right_motor.setPosition(float('inf'))
    right_motor.setVelocity(-max_speed*0.5)
    #camera_motor.setVelocity(max_speed*0.1)
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()
    range_image = lidar.getRangeImage()
    print(range_image)
    
    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    pass

# Enter here exit cleanup code.
