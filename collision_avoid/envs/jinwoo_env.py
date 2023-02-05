import airsim
import numpy as np
import math
import time
import gym
import cv2
from gym import spaces
import random
from envs.airsim_env import AirSimEnv
from PIL import Image as im
from matplotlib import pyplot as plt
from random import randrange
import tensorflow as tf
class AirSimDroneEnv(AirSimEnv):
    def __init__(self, ip_address):
        super().__init__()

        self.prev_yaw = 0
        self.state = {
            "depth": np.zeros([1,10,10,3]),
            "dyn_state": np.zeros(3),
            "position": np.zeros([1,2]),
            "collision": False
        }
        self.drone = airsim.MultirotorClient(ip=ip_address)
        self.start= time.time()
        
        #self._setup_flight()

    def _get_obs(self):
        
        self.drone_state = self.drone.getMultirotorState()

        vx = self.drone_state.kinematics_estimated.linear_velocity.x_val
        vy = self.drone_state.kinematics_estimated.linear_velocity.y_val
        yaw_rate = self.drone_state.kinematics_estimated.angular_velocity.z_val

        self.state['position'] = np.array([
            self.drone_state.kinematics_estimated.position.x_val,
            self.drone_state.kinematics_estimated.position.y_val
            ])
        self.state["dyn_state"] = self.make_batch(np.array([vx, vy, yaw_rate]))
        self.state['collision'] = self.drone.simGetCollisionInfo().has_collided
        self.state["depth"]= self.make_batch(self.lidar_data())
        
        return self.state
    
    def lidar_data(self):
        self.distance=[]
        sorted_distance=[]
        index=[]
        splited1=[]
        image_base=[]
        image_base1=[]
        image_base2=[]
        self.image_base3=[]
        self.lidarData=[]
        while True:
            if len(self.lidarData)<300:
                self.lidarData = self.drone.getLidarData().point_cloud
                if len(self.lidarData)>=300:
                    break
                else:
                    continue
        splited= np.array_split(self.lidarData, (len(self.lidarData)//3))

        for d in splited:
            splited1.append(list(d))
        for a in splited1:
            self.distance.append(math.sqrt((a[0])**2+(a[1])**2+(a[2])**2))
        sorted_distance= np.sort(self.distance)
        sorted_distance= sorted_distance[0:100]
        for b in sorted_distance:
            index.append((np.where(self.distance==b))[0])
        for c in index:
            image_base.append((splited1[int(c[0])]))
        for e in image_base:
            for f in range(3):
                image_base1.append((e[f]))
        for g in image_base1:
            if image_base1.index(g)%3==0:
                image_base2.append(2.55*g)
            elif image_base1.index(g)%3==1:
                image_base2.append(g+30)
            else:
                image_base2.append(g+10)
        for h in image_base2:
            self.image_base3.append(int(h))
        img_bytes= bytes(self.image_base3)
        image = im.frombytes("RGB", (10, 10), img_bytes)
        image_array= np.array(image)
        image_array=image_array.astype("float32")
        return image_array

    def make_batch(self, x):
        return np.expand_dims(x, axis=0)

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        x,y,z,w = airsim.utils.to_quaternion(0, 0, np.random.randint(0,360))
        loc= [[30,40], [81,57], [-43,-56], [38,112], [-20, -200], [-20, 200],[-83, -183], 
        [-233, 24], [-15, 43], [183, -189], [0,0],[-183,-189]]
        ranloc= random.choice(loc)
        x= ranloc[0]
        y= ranloc[1]
        z= -3
        position = airsim.Vector3r(x, y, z)
        pose = airsim.Pose(position)
        self.drone.simSetVehiclePose(pose, ignore_collision= True)
        
        self.drone.moveToPositionAsync(x,y,-5,3).join()

    def move(self, action):
        self.action = action
        if action[0]<0:
            action[0]=0
        vx, yaw_rate = action[0]*0.4, action[1]*10
        self.drone.moveByVelocityZBodyFrameAsync(
            vx = float(vx),
            vy = 0.0,
            z = -5.0,
            duration = 20,
            yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=float(yaw_rate))
        )     

    def get_reward(self):
        '''
        vel_x: Action is m/s, State is m/s
        vel_w: Action is degree/sec, State is rad/sec!!!
        '''
        reward_dyn=0
        reward_yaw=0
        reward_depth=0
        if self.state['collision']:
            done = 1
        else:
            done=0
        if self.action[0] <= 0:
                self.action[0] = 0

        if min(self.distance)<5 :
            
            if self.action[0]==0:
                reward_dyn= 2
                reward_yaw = -np.cos((self.action[1]*10) * np.pi/180)
                # if np.sign(self.prev_yaw) == np.sign(self.action[1]):
                #     reward_yaw= abs(self.action[1])

        else:
            reward_dyn = self.action[0]*np.cos((self.action[1]*10) * np.pi/180)/5    

        self.prev_yaw= self.action[1]
        reward = reward_dyn + reward_yaw
        # print(reward_dyn, reward)       
        return reward, done
        
    def step(self, action):
        self.move(action)
        self.obs= self._get_obs()
        self.reward, self.done= self.get_reward()
        return self.obs, self.reward, self.done

    def reset(self):
        self._setup_flight()
        return self._get_obs()
