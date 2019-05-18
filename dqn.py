import cv2
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import time
import os
import sys
import pickle
import argparse
from trex import GameState
from utils import save_image

# const
SAVE_MODEL_PERIOD = 100000
SAVE_HISTORY_PERIOD = 1000
NUMBER_OF_TRAIN_ITER = 1000000


VALID_PERIOD = 1000
NUM_VALID = 1
VALID_PRINT_PERIOD = 50
VALID_UPPER_SCORE = 10000
X_CROP_SIZE = 450


BWIMG = True
SAVE_DEBUG_IMAGE = False


# Globel best score
best_val = 0
best_itr = -1

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.number_of_actions = 3
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.number_of_actions)

        # TODO: save iteration num for future training


    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)

        return out


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)


def image_to_tensor(image):
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available():  # put on GPU if CUDA is available
        image_tensor = image_tensor.cuda()
    return image_tensor


def resize_and_bgr2gray(image):
    # image = image[0:600, 0:150]  # original
    image = image[0:X_CROP_SIZE, 0:150]  # only use a portion of it
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    if BWIMG:
        image_data[image_data > 0] = 255
    if SAVE_DEBUG_IMAGE:
        save_image(image_data, 'tmp.png')
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data


def train(model):
    global best_val, best_itr
    if not os.path.exists('loss_hist'):
        os.makedirs('loss_hist')
    if not os.path.exists('score_hist'):
        os.makedirs('score_hist')
    if not os.path.exists('pretrained_model'):
        os.makedirs('pretrained_model')

    best_val = 0
    best_itr = -1
    loss_history = []
    score_history = []

    # define Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

    # initialize mean squared error loss
    criterion = nn.MSELoss()

    # instantiate game
    game_state = GameState()

    # initialize replay memory
    replay_memory = []

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal, _ = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    # initialize epsilon value
    epsilon = model.initial_epsilon
    iteration = 0

    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, NUMBER_OF_TRAIN_ITER)

    # main infinite loop
    while iteration < NUMBER_OF_TRAIN_ITER:
        # get output from the neural network
        output = model(state)[0]

        # initialize action
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # epsilon greedy exploration
        random_action = random.random() <= epsilon
        # if random_action:
        #     print("Performed random action!")
        action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()

        action[action_index] = 1

        # get next state and reward
        image_data_1, reward, terminal, score = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)
        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        # save transition to replay memory
        replay_memory.append((state, action, reward, state_1, terminal))

        # if replay memory is full, remove the oldest transition
        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        # epsilon annealing
        epsilon = epsilon_decrements[iteration]

        # sample random minibatch
        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

        # unpack minibatch
        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

        if torch.cuda.is_available():  # put on GPU if CUDA is available
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()

        # get output for the next state
        output_1_batch = model(state_1_batch)

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))

        # extract Q-value
        q_value = torch.sum(model(state_batch) * action_batch, dim=1)

        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        # calculate loss
        loss = criterion(q_value, y_batch)
        loss_history.append(loss.cpu().detach().numpy())

        # do backward pass
        loss.backward()
        optimizer.step()

        # set state to be state_1
        state = state_1
        iteration += 1

        # validate
        if iteration % VALID_PERIOD == 0:
            print("====================")
            print("== Run validation ==")
            print("====================")
            val_scores = test(model, num_test=NUM_VALID)
            score_history.append(val_scores)

            s = np.max(np.array(val_scores))
            if best_val < s:  # if improve
                best_val = s
                best_itr = iteration
                torch.save(model, "pretrained_model/best_model.pth")
            print("Current best val score (max in {} tests): {} at itr{}".format(NUM_VALID, best_val, best_itr))
            print("====================")
            print("== End validation ==")
            print("====================")

        if iteration % SAVE_MODEL_PERIOD == 0:
            torch.save(model, "pretrained_model/current_model_" + str(iteration) + ".pth")

        if iteration % SAVE_HISTORY_PERIOD == 0:
            with open('loss_hist/loss_history.pickle', 'wb') as handle:
                pickle.dump(loss_history, handle)
            with open('score_hist/score_history.pickle', 'wb') as handle:
                pickle.dump(score_history, handle)

        print("Iter{}:: epsilon:{:<8.3}, action:{}, "
              "reward:{:4}, score:{:4}, Q max:{:<8.3}".format(iteration,
                                                  epsilon,
                                                  action_index.cpu().detach().numpy(),
                                                  reward.numpy()[0][0],
                                                  score,
                                                  np.max(output.cpu().detach().numpy())))


def test(model, num_test=10):
    game_state = GameState()

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal, _ = game_state.frame_step(action)

    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    game_cnt = 1
    scores = []
    pre_score = 0
    iteration = 0
    while True:
        iteration += 1

        # get output from the neural network
        output = model(state)[0]
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # get action
        action_index = torch.argmax(output)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()
        action[action_index] = 1

        # get next state
        image_data_1, reward, terminal, score = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        # set state to be state_1
        state = state_1
        # if iteration % VALID_PRINT_PERIOD == 0:
            # print("Test:: itr: {}, score: {}".format(iteration, score))

        if terminal or score > VALID_UPPER_SCORE:
            iteration = 0
            scores.append(pre_score)
            print("Test:: End of Game {}, score: {}\n".format(game_cnt, pre_score))
            game_cnt += 1
            if game_cnt > num_test:
                break
            game_state.__init__()
        pre_score = score

    if not os.path.exists("test_scores/"):
        os.makedirs("test_scores")

    with open('test_scores/test_score.pickle', 'wb') as handle:
        pickle.dump(scores, handle)

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="either teat/train/keeptrain", default="test", type=str)
    parser.add_argument("-m", "--model", help="path to model you want to keep training or testing",
                        default="current_model_2000000.pth", type=str)
    args = parser.parse_args()

    if args.mode == 'test':
        if torch.cuda.is_available():
            model = torch.load(args.model).eval()
        else:
            model = torch.load(args.model, map_location='cpu').eval()
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            model = model.cuda()
        # test(model, float('inf'))
        test(model)

    elif args.mode == 'train':
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')
        model = NeuralNetwork()
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            model = model.cuda()
        model.apply(init_weights)
        train(model)

    elif args.mode == 'keeptrain':
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            model = torch.load(args.model).eval()
        else:
            model = torch.load(args.model, map_location='cpu').eval()
        if torch.cuda.is_available():
            model = model.cuda()
        # model.conv1.weight.data.fill_(0.01)
        train(model)

    else:
        print("Mode: '{}' not supported".format(args.mode))


if __name__ == "__main__":
    main()
