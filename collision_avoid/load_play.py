
# SAC load and play (tf2 subclassing API version)
# coded by St.Watermelon
## 학습된 신경망 파라미터를 가져와서 에이전트를 실행시키는 파일

# 필요한 패키지 임포트
from train import A2Cagent
import tensorflow as tf
from envs.jinwoo_env import AirSimDroneEnv

def main():

    env = AirSimDroneEnv('172.17.0.1') 
    agent = A2Cagent(env)  # SAC 에이전트 객체
    

    # print(state['depth'].shape, state['dynamic_state'].shape)

    # 행동 샘플링
    state = env.reset() 
    action = agent.actor(state['depth'], state['dyn_state'])
    print(action)
    agent.load_weights('./collision_avoid/save_weights/')  # 신경망 파라미터 가져옴
    
    for _ in range(100000000):
        time = 0
        state = env.reset()  # 환경을 초기화하고, 초기 상태 관측
        while True:
            env.render()
            # 행동 계산
            action = agent.get_action(state['depth'], state['dyn_state'])

            state, reward, done = env.step(action)
            time += 1

            print('Time: ', time, 'Reward: ', reward)

            if done:
                break

    env.close()


if __name__=="__main__":
    main()