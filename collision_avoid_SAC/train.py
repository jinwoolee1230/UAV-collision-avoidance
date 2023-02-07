import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Dense, Lambda, Flatten, Concatenate, Conv2D, MaxPooling2D
from keras.optimizers import Adam

import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

import tensorflow_probability as tfp

from replaybuffer import ReplayBuffer


## 액터 신경망
class Actor(Model):

    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()

        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = [1e-1, 1.0]

        self.conv1 = Conv2D(kernel_size=7, filters=32, strides=1, padding='same', activation='relu')
        self.maxpool1 = MaxPooling2D()
        
        self.conv2 = Conv2D(kernel_size=3, filters=16, strides=1, padding='same', activation='relu')
        self.maxpool2 = MaxPooling2D() #16, 40, 16 = 560 * 16 ~= 5600

        self.conv3 = Conv2D(kernel_size=3, filters=1, strides=1, padding='same', activation='relu')
        self.maxpool3 = MaxPooling2D() #8, 20, 1 = 8*20 = 160
        
        self.flatten1 = Flatten()
        self.depth_h1 = Dense(80, activation='relu')
        self.depth_h2 = Dense(40, activation='relu')
        
        #dyn
        self.flatten2 = Flatten() # 1,12 or 4,1,12
        self.dyn_h1 = Dense(30, activation='relu')
        self.dyn_h2 = Dense(40, activation='relu')

        
        self.concat = Concatenate()
        
        self.h1 = Dense(80, activation='relu')
        self.h2 = Dense(40, activation='relu')
        self.h3 = Dense(20, activation='relu')
        self.mu = Dense(action_dim, activation='relu')
        self.std = Dense(action_dim, activation='softplus')


    def call(self, state1, state2):
        depth, dyn_state = state1, state2
        #print (state[0].shape)
        #print (state[1].shape)

        depth_features = self.conv1(depth)
        depth_features = self.maxpool1(depth_features)
        depth_features = self.conv2(depth_features)
        depth_features = self.maxpool2(depth_features)
        depth_features = self.conv3(depth_features)
        depth_features = self.maxpool3(depth_features)
        
        depth_features = self.flatten1(depth_features)
        depth_features = self.depth_h1(depth_features)
        depth_features = self.depth_h2(depth_features)
        
        #dyn
        dyn_features = self.flatten2(dyn_state) # 1,12 or 4,1,12
        dyn_features = self.dyn_h1(dyn_features)
        dyn_features = self.dyn_h2(dyn_features)        
        #print(f"shape1 : {depth_features.shape}, shape2: {dyn_features.shape}")
        concat_state = self.concat([depth_features, dyn_features])
        x = self.h1(concat_state)
        x = self.h2(x)
        x = self.h3(x)
        mu = self.mu(x)
        std = self.std(x)

        # 평균값을 [-action_bound, action_bound] 범위로 조정
        mu = Lambda(lambda x: x*self.action_bound)(mu)
        # 표준편차 클래핑
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])

        return mu, std

    ## 행동을 샘플링하고 log-pdf 계산
    def sample_normal(self, mu, std):
        normal_prob = tfp.distributions.Normal(mu, std)
        action = normal_prob.sample()
        action = tf.clip_by_value(action, -self.action_bound, self.action_bound)
        log_pdf = normal_prob.log_prob(action)
        log_pdf = tf.reduce_sum(log_pdf, 1, keepdims=True)

        return action, log_pdf


## 크리틱 신경망
class Critic(Model):

    def __init__(self):
        super(Critic, self).__init__()

        self.conv1 = Conv2D(kernel_size=7, filters=32, strides=1, padding='same', activation='relu')
        self.maxpool1 = MaxPooling2D()

        self.conv2 = Conv2D(kernel_size=3, filters=16, strides=1, padding='same', activation='relu')
        self.maxpool2 = MaxPooling2D() #16, 40, 16 = 560 * 16 ~= 5600

        self.conv3 = Conv2D(kernel_size=3, filters=1, strides=1, padding='same', activation='relu')
        self.maxpool3 = MaxPooling2D() #8, 20, 1 = 8*20 = 160
        
        self.flatten1 = Flatten()
        self.depth_h1 = Dense(80, activation='relu')
        self.depth_h2 = Dense(40, activation='relu')
        
        #dyn
        self.flatten2 = Flatten() # 1,12 or 4,1,12
        self.dyn_h1 = Dense(30, activation='relu')
        self.dyn_h2 = Dense(40, activation='relu')

        
        self.concat = Concatenate()
        
        self.h1 = Dense(80, activation='relu')
        self.h2 = Dense(40, activation='relu')
        self.h3 = Dense(20, activation='relu')

        
        self.x1 = Dense(32, activation='relu')
        self.a1 = Dense(32, activation='relu')
        self.concat2 = Concatenate(axis=-1)
        self.h4 = Dense(32, activation='relu')
        self.h5 = Dense(16, activation='relu')
        self.q = Dense(1, activation='linear')
        
        

    def call(self, state_action1, state_action2):

        depth, dyn_state = state_action1, state_action2

        depth_features = self.conv1(depth)
        depth_features = self.maxpool1(depth_features)
        depth_features = self.conv2(depth_features)
        depth_features = self.maxpool2(depth_features)
        depth_features = self.conv3(depth_features)
        depth_features = self.maxpool3(depth_features)
        
        depth_features = self.flatten1(depth_features)
        depth_features = self.depth_h1(depth_features)
        depth_features = self.depth_h2(depth_features)
        
        #dyn
        dyn_features = self.flatten2(dyn_state) # 1,12 or 4,1,12
        dyn_features = self.dyn_h1(dyn_features)
        dyn_features = self.dyn_h2(dyn_features)        

        concat_state = self.concat([depth_features, dyn_features])
        x = self.h1(concat_state)
        x = self.h2(x)
        x = self.h3(x)
        q = self.q(x)
        return q


## SAC 에이전트
class SACagent(object):

    def __init__(self, env):

        # 하이퍼파라미터
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.BUFFER_SIZE = 20000
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        self.TAU = 0.001
        self.ALPHA = 0.5
        # 환경
        self.env = env
        # 상태변수 차원
        self.state_dim = 4
        # 행동 차원
        self.action_dim = 2
        # 행동의 최대 크기
        self.action_bound = 5

        self.std_bound = [1e-2, 1.0]

        # 액터 신경망 및 Q1, Q2 타깃 Q1, Q2 신경망 생성
        self.actor = Actor(self.action_dim, self.action_bound)

        self.critic_1 = Critic()
        self.target_critic_1 = Critic()

        self.critic_2 = Critic()
        self.target_critic_2 = Critic()

        # 옵티마이저
        self.actor_opt = Adam(self.ACTOR_LEARNING_RATE)
        self.critic_1_opt = Adam(self.CRITIC_LEARNING_RATE)
        self.critic_2_opt = Adam(self.CRITIC_LEARNING_RATE)

        # 리플레이 버퍼 초기화
        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        # 에피소드에서 얻은 총 보상값을 저장하기 위한 변수
        self.save_epi_reward = []


    ## 행동 샘플링
    def get_action(self, state1, state2):
        mu, std = self.actor(state1, state2)
        std= np.clip(std, self.std_bound[0], self.std_bound[1])
        action, _ = self.actor.sample_normal(mu, std)
        return action[0]


    ## 신경망의 파라미터값을 타깃 신경망으로 복사
    def update_target_network(self, TAU):
        phi_1 = self.critic_1.get_weights()
        phi_2 = self.critic_2.get_weights()
        target_phi_1 = self.target_critic_1.get_weights()
        target_phi_2 = self.target_critic_2.get_weights()
        for i in range(len(phi_1)):
            target_phi_1[i] = TAU * phi_1[i] + (1 - TAU) * target_phi_1[i]
            target_phi_2[i] = TAU * phi_2[i] + (1 - TAU) * target_phi_2[i]
        self.target_critic_1.set_weights(target_phi_1)
        self.target_critic_2.set_weights(target_phi_2)


    ## Q1, Q2 신경망 학습
    def critic_learn(self, states1, states2, actions, q_targets):
        with tf.GradientTape() as tape:
            q_1 = self.critic_1([states1, states2, actions], training=True)
            loss_1 = tf.reduce_mean(tf.square(q_1-q_targets))

        grads_1 = tape.gradient(loss_1, self.critic_1.trainable_variables)
        self.critic_1_opt.apply_gradients(zip(grads_1, self.critic_1.trainable_variables))

        with tf.GradientTape() as tape:
            q_2 = self.critic_2([states1, states2, actions], training=True)
            loss_2 = tf.reduce_mean(tf.square(q_2-q_targets))

        grads_2 = tape.gradient(loss_2, self.critic_2.trainable_variables)
        self.critic_2_opt.apply_gradients(zip(grads_2, self.critic_2.trainable_variables))


    ## 액터 신경망 학습
    def actor_learn(self, states1, states2):
        with tf.GradientTape() as tape:
            mu, std = self.actor(states1, states2, training=True)
            actions, log_pdfs = self.actor.sample_normal(mu, std)
            log_pdfs = tf.squeeze(log_pdfs, 1)
            soft_q_1 = self.critic_1(states1, states2, actions)
            soft_q_2 = self.critic_2(states1, states2, actions)
            soft_q = tf.math.minimum(soft_q_1, soft_q_2)

            loss = tf.reduce_mean(self.ALPHA * log_pdfs - soft_q)

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))


    ## 시간차 타깃 계산
    def q_target(self, rewards, q_values, dones):
        y_k = np.asarray(q_values)
        for i in range(q_values.shape[0]): # number of batch
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.GAMMA * q_values[i]
        return y_k


    ## 신경망 파라미터 로드
    def load_weights(self, path):
        self.actor.load_weights(path + 'pendulum_actor_2q.h5')
        self.critic_1.load_weights(path + 'pendulum_critic_12q.h5')
        self.critic_2.load_weights(path + 'pendulum_critic_22q.h5')


    ## 에이전트 학습
    def train(self, max_episode_num):

        # 타깃 신경망 초기화
        self.update_target_network(1.0)

        # 에피소드마다 다음을 반복
        for ep in range(int(max_episode_num)):

            # 에피소드 초기화
            time, episode_reward, done = 0, 0, False
            # 환경 초기화 및 초기 상태 관측
            state = self.env.reset()

            while not done:
                # 환경 가시화
                #self.env.render()
                # 행동 샘플링
                action = self.get_action(state["depth"], state["dyn_state"])                # 행동 범위 클리핑
                action = np.clip(action, -self.action_bound, self.action_bound)
                # 다음 상태, 보상 관측
                next_state, reward, done, _= self.env.step(action)
                # 학습용 보상 설정
                train_reward = reward
                # 리플레이 버퍼에 저장
                self.buffer.add_buffer(state["depth"], state["dyn_state"], action, reward, next_state["depth"], next_state["dyn_state"], done)

                # 리플레이 버퍼가 일정 부분 채워지면 학습 진행
                if self.buffer.buffer_count() > 1000:

                    # 리플레이 버퍼에서 샘플 무작위 추출
                    states_depth, states_dyn, actions, rewards, next_states_depth, next_states_dyn, dones = self.buffer.sample_batch(self.BATCH_SIZE)

                    # Q 타깃 계산
                    next_mu, next_std = self.actor(tf.convert_to_tensor(next_states_depth, next_states_dyn))
                    prev_nd= next_states_depth
                    prev_ndy=next_states_dyn

                    next_actions, next_log_pdf = self.actor.sample_normal(next_mu, next_std)

                    target_qs_1 = self.target_critic_1([next_states_depth, next_states_dyn, next_actions])
                    target_qs_2 = self.target_critic_2([next_states_depth, next_states_dyn, next_actions])
                    target_qs = tf.math.minimum(target_qs_1, target_qs_2)

                    target_qi = target_qs - self.ALPHA * next_log_pdf

                    # TD 타깃 계산
                    y_i = self.q_target(rewards, target_qi.numpy(), dones)

                    # Q1, Q2 신경망 업데이트
                    self.critic_learn(tf.convert_to_tensor(states_depth, states_dyn, dtype=tf.float32),
                                      tf.convert_to_tensor(actions, dtype=tf.float32),
                                      tf.convert_to_tensor(y_i, dtype=tf.float32))

                    # 액터 신경망 업데이트
                    self.actor_learn(tf.convert_to_tensor(states_depth, states_dyn, dtype=tf.float32))

                    # 타깃 신경망 업데이트
                    self.update_target_network(self.TAU)

                # 다음 스텝 준비
                state = next_state
                episode_reward += reward
                time += 1

            # 에피소드마다 결과 보상값 출력
            print('Episode: ', ep+1, 'Time: ', time, 'Reward: ', episode_reward)
            self.save_epi_reward.append(episode_reward)


            # 에피소드마다 신경망 파라미터를 파일에 저장
            self.actor.save_weights("./collision_avoid_SAC/save_weights/pendulum_actor_2q.h5")
            self.critic_1.save_weights("./collision_avoid_SAC/save_weights/pendulum_critic_12q.h5")
            self.critic_2.save_weights("./collision_avoid_SAC/save_weights/pendulum_critic_22q.h5")

        # 학습이 끝난 후, 누적 보상값 저장
        np.savetxt('.collision_avoid_SAC/save_weights/pendulum_epi_reward_2q.txt', self.save_epi_reward)
        print(self.save_epi_reward)


    ## 에피소드와 누적 보상값을 그려주는 함수
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()