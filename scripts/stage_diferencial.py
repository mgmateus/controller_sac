#! /usr/bin/env python3
import rospy 
import math
import numpy as np

from std_srvs.srv import Empty
#from trajectory_msgs.msg import Transform
from geometry_msgs.msg import Twist, Transform
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

class Robot:
	def __init__(self, target=(0.0, 0.0), linear_vel=(0.0, 0.5), angular_vel=(-0.25, 0.25)):
		rospy.Subscriber("/base_pose_ground_truth", Odometry, self.callback_odom)
		rospy.Subscriber("/base_scan", LaserScan, self.callback_lidar)

		
		self.__linear_vel = linear_vel
		self.__angular_vel = angular_vel
		self.__vel = Twist() 
		self.__yaw = 0
		self.__pub_vel= rospy.Publisher('/cmd_vel', Twist, queue_size=10)

		self.__target = target
		self.__min_distance = 0.1

		self.__odom = Odometry()
		self.__lidar = []
		self.__pose = (-3.0, -1.0)
		self.__distance = math.sqrt((self.__pose[0]-self.__target[0])**2 + (self.__pose[1]-self.__target[1])**2)
		
	@property
	def target(self):
		return self.__target
	
	@target.setter
	def target(self, target):
		self.__target = target

	@property
	def min_distance(self):
		return self.__min_distance

	@min_distance.setter
	def min_distance(self, val):
		self.__min_distance = val

	@property
	def distance(self):
		return self.__distance
	
	@property
	def pose(self):
		return self.__pose
	
	def callback_odom(self, msg):
		self.__odom = msg
		self.__pose = (msg.pose.pose.position.x, msg.pose.pose.position.y)

	def callback_lidar(self, scan):
		scan_range = []

		for i in range(len(scan.ranges)):
			if scan.ranges[i] == float('Inf'):
				scan_range.append(30)
			elif np.isnan(scan.ranges[i]):
				scan_range.append(0.1)
			else:
				scan_range.append(scan.ranges[i])

		self.__lidar = scan_range[270:811]

	def reset_position(self):
		try:
			service = rospy.ServiceProxy("/reset_positions", Empty)
			rospy.wait_for_service("/reset_positions")
			service()

		except rospy.ServiceException as e:
			print ('Service call failed: %s' % e)

	def collision(self):
		return True if min(self.__lidar) <= 0.2 else False
	
	def quaternion_to_euler(self, x, y, z, w):

		t0 = +2.0 * (w * x + y * z)
		t1 = +1.0 - 2.0 * (x * x + y * y)
		roll = math.atan2(t0, t1)
		t2 = +2.0 * (w * y - z * x)
		t2 = +1.0 if t2 > +1.0 else t2
		t2 = -1.0 if t2 < -1.0 else t2
		pitch = math.asin(t2)
		t3 = +2.0 * (w * z + x * y)
		t4 = +1.0 - 2.0 * (y * y + z * z)
		yaw = math.atan2(t3, t4)
		return [yaw, pitch, roll]
	
	def yaw(self):
		trans = Transform()
		trans.translation.x = self.__odom.pose.pose.position.x
		trans.translation.y = self.__odom.pose.pose.position.y
		trans.translation.z = self.__odom.pose.pose.position.z
		trans.rotation.x = self.__odom.pose.pose.orientation.x
		trans.rotation.y = self.__odom.pose.pose.orientation.y
		trans.rotation.z = self.__odom.pose.pose.orientation.z
		trans.rotation.w = self.__odom.pose.pose.orientation.w

		yaw, _, _ = self.quaternion_to_euler(trans.rotation.x, trans.rotation.y, trans.rotation.z, trans.rotation.w)
		return yaw
	
	def theta(self):
		heading = np.arctan2(self.__target[1] - self.__pose[1], self.__target[0] - self.__pose[0]) - self.yaw()
		if heading > math.pi:
			heading -= 2 * math.pi

		elif heading < -math.pi:
			heading += 2 * math.pi
		return heading
	

	def get_state(self, action):
		self.__vel.linear.x = np.clip(action[0], self.__linear_vel[0], self.__linear_vel[1])
		self.__vel.angular.z = np.clip(action[1], self.__angular_vel[0], self.__angular_vel[1])
		self.__pub_vel.publish(self.__vel)


		self.__distance = math.sqrt((self.__pose[0]-self.__target[0])**2 + (self.__pose[1]-self.__target[1])**2)
		done = True if self.__distance <= self.__min_distance else False

		theta = self.theta()
		
		observation = [self.__pose[0], self.__pose[1], self.__distance, theta] + self.__lidar
		
		return observation, done


	def __str__(self) -> str:
		p = "------------------\n"
		p += "Position: ({:.2f}, {:.2f}) =====> Target: {}\n".format(self.__pose[0] , self.__pose[1], self.__target)
		p += "Distance: {:.2f}\n".format(self.__distance)
		p += "------------------\n"
		return p



