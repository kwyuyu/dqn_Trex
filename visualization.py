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
from utils import save_image, save_immediate_image

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


    def forward(self, x, iteration):
        out = self.conv1(x)
        # save_immediate_image(out, 'iter%d_conv1.png' % (iteration))
        out = self.relu1(out)
        out = self.conv2(out)
        # save_immediate_image(out, 'iter%d_conv2.png' % (iteration))
        out = self.relu2(out)
        out = self.conv3(out)
        # save_immediate_image(out, 'iter%d_conv3.png' % (iteration))
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



def test(model, num_test=1):
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
    with torch.no_grad():
        while True:
            iteration += 1

            # get output from the neural network
            output = model(state, iteration)[0]
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

            if terminal or score > VALID_UPPER_SCORE:
                iteration = 0
                scores.append(pre_score)
                print("Test:: End of Game {}, score: {}\n".format(game_cnt, pre_score))
                game_cnt += 1
                if game_cnt > num_test:
                    break
                game_state.__init__()
            pre_score = score

    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="path to model you want to keep training or testing",
                        default="current_model_2000000.pth", type=str)
    args = parser.parse_args()

    if torch.cuda.is_available():
        model = torch.load(args.model).eval()
    else:
        model = torch.load(args.model, map_location='cpu').eval()
    if torch.cuda.is_available():  # put on GPU if CUDA is available
        model = model.cuda()
    test(model)



if __name__ == "__main__":
    main()
