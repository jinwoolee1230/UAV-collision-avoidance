import numpy as np
from collections import deque
import random

class ReplayBuffer(object):
    """
    Reply Buffer
    """
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.count = 0

    ## 버퍼에 저장
    def add_buffer(self, state_depth, state_dyn, action, reward, next_state_depth, next_state_dyn, done):
        transition = (state_depth, state_dyn, action, reward, next_state_depth, next_state_dyn, done)

        # 버퍼가 꽉 찼는지 확인
        if self.count < self.buffer_size:
            self.buffer.append(transition)
            self.count += 1
        else: # 찼으면 가장 오래된 데이터 삭제하고 저장
            self.buffer.popleft()
            self.buffer.append(transition)

    ## 버퍼에서 데이터 무작위로 추출 (배치 샘플링)
    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        # 상태, 행동, 보상, 다음 상태별로 정리
        states_depth = np.asarray([i[0] for i in batch])
        states_dyn = np.asarray([i[1] for i in batch])

        actions = np.asarray([i[2] for i in batch])
        rewards = np.asarray([i[3] for i in batch])
        next_states_depth = np.asarray([i[4] for i in batch])
        next_states_dyn = np.asarray([i[5] for i in batch])
        dones = np.asarray([i[6] for i in batch])
        return states_depth, states_dyn, actions, rewards, next_states_depth, next_states_dyn, dones


    ## 버퍼 사이즈 계산
    def buffer_count(self):
        return self.count


    ## 버퍼 비움
    def clear_buffer(self):
        self.buffer = deque()
        self.count = 0