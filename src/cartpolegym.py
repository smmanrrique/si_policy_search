# Controling a cart and pole system using REINFORCE
#
# Author: Ruben Martinez Cantin <rmcantin@unizar.es>.
#         Universidad de Zaragoza
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Ruben Martinez Cantin and the University of Zaragoza.


import numpy as np
import gym


class PGAgent:
    '''Abstract class for Policy Gradient Agents'''

    def policy(self, state):
        raise NotImplementedError

    def gradient_log_policy(self, state, action):
        return 0

    def update_policy(self, grads, rewards):
        pass

    def update_learning_rate(self, iteration):
        pass


class RandomAgent(PGAgent):
    '''Agent that always follows a random policy.'''

    def policy(self, state):
        '''Returns a list or numpy vector with the probability for each action.
        All the returned values must add to 1.
        '''
        if state[2] < 0:
            return [1, 0]
        else:
            return [0, 1]
        return [0.4 , 0.6]


class NaiveAgent(PGAgent):
    '''Agent that uses a simple bang-bang controller.'''
    def policy(self, state):
        return [0.6 , 0.4]  

class LinearAgent(PGAgent):
    '''Agent that uses a softmax policy trained using REINFORCE.'''

    def __init__(self, n_states, n_actions):
        '''Define all the elements that you might need.

        Instance variables that you need to define:
        - Policy parameters
        - Learning rate

        Should initialize the policy parameters using samples from a
        normal distribution Check numpy.random.randn

        There should be a parameter for each feature (see policy). You
        might able to reduce the code size by vectorizing the
        operations.
        '''
        # *** YOUR CODE HERE ***
        poli = 0+np.random.randn()+1



    def policy(self, state):
        '''Returns a list or numpy vector with the probabilities for each action.

        For the features, you should directly use the state multiplied
        by the indicator function of the action. Remember: for the
        "left" action, indicator=1 if action=left and indicator=0 if
        action=right.

        Finally, you should use the softmax expresion for the
        probabilities, as seen in the class slides.

        '''
        # *** YOUR CODE HERE ***

    def gradient_log_policy(self, state, action):
        '''Returns the gradient of the log of the policy given the last state
        and action.

        You might use the datatype that you find more suitable to
        update the policy in the next method.

        Remember: The gradient of the log for the softmax policy is
        the features given the selected action minus the sum over all
        action of the features times the probability of each action.

        '''
        # *** YOUR CODE HERE ***

    def update_policy(self, grads, rewards):
        '''Update the parameters of the policy given an episode.

        grads and rewards contain a list of all the policy gradients
        and rewards collected at each timestep of the episode. You
        should use that to update the parameters of the policy as many
        times as possible.
        '''
        # *** YOUR CODE HERE ***

    def update_learning_rate(self, iteration):
        '''Update the learning rate after each episode. (OPTIONAL)

        Initially, ignore this method and use a constant learning
        rate. Once your policy and gradient are working, you can
        optimize the learning rate.

        This method is called after each episode to update the
        learning rate. This can be used to have a diminishing learning
        rate and follow the Robbins-Monrow conditions. 

        Recomended values for the learning rate are, for example:

        - alpha = 1/i
        - alpha = 1/(i + c)

        where i is the number of episode (iteration) and c is a
        constant. The constant can be used to avoid very large rates
        in the first iterations.
        '''
        # *** YOUR CODE HERE ***


class CartPole:
    '''Class to configure the Gym environment and run the training and
    testing scenarios.

    You should not need to change anything from this code, except the
    environment in the agent which are defined in the __init__
    method. If you decide to change anything, add a commentary and
    explain the changes in the report.

    '''

    def __init__(self):

        # **** CHANGE THIS **********************************
        self.env = gym.make("CartPole-v1")
        # self.env = gym.make("CartPole-v0")
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        #   Select the agent here
        # self.agent = NaiveAgent()
        self.agent = RandomAgent()
        self.agent = LinearAgent(self.n_states, self.n_actions)
        # ****************************************************

        self.episode_rewards = []
        self.test_episode_rewards = []
        self.render = False

    def run_episode(self):
        '''Run an iteration of the REINFORCE algorithm.

        - Collect all data from a single episode
        - Compute the policy gradient at each iteration
        - Update the policy parameters at the end of the episode
        '''

        # Initial state
        state = self.env.reset()
        grads, rewards, actions = [], [], []
        score = 0

        while True:
            if self.render:
                self.env.render()

            # Select action and find new step
            action_probabilities = self.agent.policy(state)
            action = np.random.choice(self.n_actions, p=action_probabilities)
            next_state, reward, done, info = self.env.step(action)

            # Compute gradient
            grad = self.agent.gradient_log_policy(state, action)

            # Keep a record of the grads, rewards and actions of this
            # episode
            grads.append(grad)
            rewards.append(reward)
            actions.append(action)

            # We use this to report the total reward.
            score += reward

            # Before the next iteration, we update the state
            state = next_state

            if done:
                # If the episode have finished, update the policy
                # parameters
                self.agent.update_policy(grads, rewards)
                break

        # Append for logging and print
        self.episode_rewards.append(score)
        return score

    def run_test_episode(self):
        '''Run an episode only for testing (check total reward).

        - Collect all data from a single episode
        - The policy does not change.
        '''

        # Initial state
        state = self.env.reset()
        score = 0

        while True:
            if self.render:
                self.env.render()

            # Select action and find new step
            action_probabilities = self.agent.policy(state)
            action = np.argmax(action_probabilities)
            next_state, reward, done, info_ = self.env.step(action)

            score += reward

            # Dont forget to update your old state to the new state
            state = next_state

            if done:
                break

        # Append for logging and print
        self.test_episode_rewards.append(score)

    def train(self, num_episodes=1000):
        ''' Run multiple training episodes. '''

        for i in range(num_episodes):
            self.agent.update_learning_rate(i)
            score = self.run_episode()
            score = self.run_episode()

            print("EP: " + str(i) + " Score: " + str(score) +
                  "         ", end="\r", flush=False)

    def test(self, num_tests=100):
        ''' Run multiple testing episodes. '''

        for i in range(num_tests):
            score = self.run_test_episode()

        print("Average Test: ", sum(self.test_episode_rewards) / num_tests)

    def close(self):
        self.env.close()


if __name__ == "__main__":
    cart = CartPole()

    # *** Uncomment this if you want to render during training *****
    cart.render = True
    cart.train()

    print("Training done")

    # *** Uncomment this if you want to render during testing *****
    cart.render = True
    cart.test(10)

    cart.close()

    # If you want to plot the score evolution during training and testing, you can use this:
    
    import matplotlib.pyplot as plt
    plt.plot(cart.episode_rewards)
    plt.plot(cart.test_episode_rewards, 'r')
    plt.legend(['Training', 'Testing'])
    plt.show()
    
    # In some systems, there might be a conflict between the Gym
    # render engine and matplotlib. You can avoid that by turning off
    # the render when you want to use matplotlib.
