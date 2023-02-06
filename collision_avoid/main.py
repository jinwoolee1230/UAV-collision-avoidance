from train import A2Cagent 
import gym
from envs.jinwoo_env import AirSimDroneEnv
import numpy as np
import sys
import time
np.set_printoptions(threshold=sys.maxsize)
def main():

    max_episode_num = 1000   # 최대 에피소드 설정
    env = AirSimDroneEnv ("172.17.0.1")
    #while True:
    #        env.drone.moveByVelocityAsync(3,0,0,3)
    #        env.lidar_data()
    #        time.sleep(0.3)

    agent = A2Cagent(env)   # A2C 에이전트 객체
    
    #agent.actor.load_weights('actor.h5')
    #agent.critic.load_weights('critic.h5')
    # 학습 진행
    agent.train(max_episode_num)

    # 학습 결과 도시
    agent.plot_result()

if __name__=="__main__":
    main()