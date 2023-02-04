import tensorflow as tf

from keras.models import Model
from keras.layers import Input, Dense, Lambda, Flatten, Concatenate, Conv2D, MaxPooling2D
from keras.optimizers import Adam

import datetime
import numpy as np
import matplotlib.pyplot as plt

## A2C CNN 액터 신경망

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



## A2C 크리틱 신경망
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

## A2C 에이전트 클래스
class A2Cagent(object):

    def __init__(self, env):
        
        # 하이퍼파라미터
        self.GAMMA = 0.95
        self.BATCH_SIZE = 32
        self.ACTOR_LEARNING_RATE = 0.0001
        self.CRITIC_LEARNING_RATE = 0.001
        # 환경
        self.env = env
        # 상태변수 차원
        self.state_dim = 4
        # 행동 차원
        self.action_dim = 2
        # 행동의 최대 크기
        self.action_bound = 5
        # 표준편차의 최솟값과 최대값 설정
        self.std_bound = [1e-2, 1.0]

        # 액터 신경망 및 크리틱 신경망 생성
        self.actor = Actor(self.action_dim, self.action_bound)
        self.critic = Critic()
        #self.actor.build(input_shape=(None, self.state_dim))
        #self.critic.build(input_shape=(None, self.state_dim))

        #self.actor.summary()
        #elf.critic.summary()

        # 옵티마이저 설정
        self.actor_opt = Adam(self.ACTOR_LEARNING_RATE)
        self.critic_opt = Adam(self.CRITIC_LEARNING_RATE)

        # 에프소드에서 얻은 총 보상값을 저장하기 위한 변수
        self.save_epi_reward = []

        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = 'logs/' + self.current_time + '/train'
        self.train_summary_writer = tf.summary.create_file_writer(self.log_dir)


    ## 로그-정책 확률밀도함수
    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)


    ## 액터 신경망에서 행동 샘플링
    def get_action(self, state1, state2):
        mu, std = self.actor(state1, state2)
        std= np.clip(std, self.std_bound[0], self.std_bound[1])
        #print(f"mu:{mu.shape}, std:{std.shape}")
        action= np.random.normal(mu, std, size=(1,2))#self.action_dim)
        return action[0]
        


    ## 액터 신경망 학습
    def actor_learn(self, state1, state2, actions, advantages):
        states_depth= state1
        states_dyn= state2

        with tf.GradientTape() as tape:
            # 정책 확률밀도함수
            mu, std = self.actor(states_depth, states_dyn, training=True)
            log_policy_pdf = self.log_pdf(mu, std, actions)

            # 손실함수
            loss_policy = log_policy_pdf * advantages
            loss = tf.reduce_sum(-loss_policy)
        with self.train_summary_writer.as_default():
            tf.summary.scalar('actor_loss', loss, step=self.time)

        # 그래디언트
        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))

    ## 크리틱 신경망 학습
    def critic_learn(self, states1, states2, td_targets):
        states_depth= states1
        states_dyn= states2
        with tf.GradientTape() as tape:
            td_hat = self.critic(states_depth, states_dyn, training=True)
            loss = tf.reduce_mean(tf.square(td_targets-td_hat))
        with self.train_summary_writer.as_default():
                    tf.summary.scalar('critic_loss', loss, step=self.time)

        grads= tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))

    ## 시간차 타깃 계산
    def td_target(self, rewards, next_v_values, dones):
        y_i = np.zeros(next_v_values.shape)
        for i in range(next_v_values.shape[0]):
            if dones[i]:
                y_i[i] = rewards[i]
            else:
                y_i[i] = rewards[i] + self.GAMMA * next_v_values[i]
        return y_i


    ## 신경망 파라미터 로드
    def load_weights(self, path):
        self.actor.load_weights(path + 'actor.h5')
        self.critic.load_weights(path + 'critic.h5')


    ## 배치에 저장된 데이터 추출
    def unpack_batch(self, batch):
        unpack = batch[0]
        for idx in range(len(batch)-1):
            unpack = np.append(unpack, batch[idx+1], axis=0)

        return unpack


    ## 에이전트 학습
    def train(self, max_episode_num):

        # 에피소드마다 다음을 반복
        for ep in range(int(max_episode_num)):

            # 배치 초기화
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = [], [], [], [], []
            batch_state1=[]
            batch_next_state1=[]
            # 에피소드 초기화
            self.time, episode_reward, done = 0, 0, False
            # 환경 초기화 및 초기 상태 관측
            state= self.env.reset()

            while not done:

                # 행동 샘플링
                action = self.get_action(state["depth"], state["dyn_state"])
                # 행동 범위 클리핑
                action = np.clip(action, -self.action_bound, self.action_bound)
                # 다음 상태, 보상 관측
                next_state, reward, done= self.env.step(action)
                #TQDM CODE
                # shape 변환
                #action = np.reshape(action, [1, self.action_dim])
                #reward = np.reshape(reward, [1, 1])
                #next_state = np.reshape(next_state, [1, self.state_dim])
                #done = np.reshape(done, [1, 1])
                # 학습용 보상 계산
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('Step_reward', reward, step=self.time)
                train_reward = reward

                # 배치에 저
                batch_state.append(state["depth"])
                batch_state1.append(state["dyn_state"])
                batch_action.append([action])
                batch_reward.append([train_reward])
                batch_next_state.append(next_state["depth"])
                batch_next_state1.append(next_state["dyn_state"])
                batch_done.append([done])


                # 배치가 채워질 때까지 학습하지 않고 저장만 계속
                if len(batch_state) < self.BATCH_SIZE and len(batch_state1) < self.BATCH_SIZE:
                    # 상태 업데이트
                    state = next_state
                    episode_reward += reward
                    self.time += 1
                    continue
                # 배치가 채워지면 학습 진행
                # 배치에서 대이터 추출'''
                train_rewards = self.unpack_batch(batch_reward)
                states = self.unpack_batch(batch_state)
                states1 = self.unpack_batch(batch_state1)
                actions = self.unpack_batch(batch_action)
                next_states = self.unpack_batch(batch_next_state)
                next_states1 = self.unpack_batch(batch_next_state1)
                
                dones = self.unpack_batch(batch_done)

                # 배치 비움
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = [], [], [], [], []
                batch_next_state1=[]
                batch_state1=[]
                # 시간차 타깃 계산
                next_v_values = self.critic(tf.convert_to_tensor(next_states, dtype=tf.float32),
                                            tf.convert_to_tensor(next_states1, dtype=tf.float32))
                td_targets = self.td_target(train_rewards, next_v_values.numpy(), dones)

                # 크리틱 신경망 업데이트
                self.critic_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                    tf.convert_to_tensor(states1, dtype=tf.float32),
                                  tf.convert_to_tensor(td_targets, dtype=tf.float32))

                # 어드밴티지 계산
                v_values = self.critic(tf.convert_to_tensor(states, dtype=tf.float32),
                                        tf.convert_to_tensor(states1, dtype=tf.float32))
                next_v_values = self.critic(tf.convert_to_tensor(next_states, dtype=tf.float32),
                                            tf.convert_to_tensor(next_states1, dtype=tf.float32))
                advantages = train_rewards + self.GAMMA * next_v_values - v_values

                # 액터 신경망 업데이트
                self.actor_learn(tf.convert_to_tensor(states, dtype=tf.float32),
                                tf.convert_to_tensor(states1, dtype=tf.float32),
                                 tf.convert_to_tensor(actions, dtype=tf.float32),
                                 tf.convert_to_tensor(advantages, dtype=tf.float32))

                # 상태 업데이트
                state = next_state
                episode_reward += reward

            # 에피소드마다 결과 출력
            if self.time>1:
                print('Episode: ', ep+1, 'Time: ', self.time, 'Reward: ', episode_reward)
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('Episode_reward', episode_reward, step=ep)
                self.save_epi_reward.append(episode_reward)
            else:
                ep-=1


            # 에피소드 10번마다 신경망 파라미터를 파일에 저장
            if ep % 10 == 0:
                self.actor.save_weights("./collision_avoid/save_weights/actor.h5")
                self.critic.save_weights("./collision_avoid/save_weights/critic.h5")

        # 학습이 끝난 후, 누적 보상값 저장
        np.savetxt('./collision_avoid/save_weights/reward.txt', self.save_epi_reward)
        print(self.save_epi_reward)


    ## 에피소드와 누적 보상값을 그려주는 함수
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()