from train import SACagent
import gym
from envs.jinwoo_env import AirSimDroneEnv
import numpy as np
import sys
import time
np.set_printoptions(threshold=sys.maxsize)
def main():

    max_episode_num = 1000   # 최대 에피소드 설정
    # env = AirSimDroneEnv ("127.0.0.1")
    env = AirSimDroneEnv ("10.74.23.239")
    #while True:
    #        env.drone.moveByVelocityAsync(3,0,0,3)
    #        env.lidar_data()
    #        time.sleep(0.3)

    agent = SACagent(env)   # A2C 에이전트 객체

    
    # 학습 진행
    agent.train(max_episode_num)

    # 학습 결과 도시
    agent.plot_result()

if __name__=="__main__":
    main()