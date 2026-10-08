import numpy as np
import matplotlib.pyplot as plt
import os
import sys

class MDP:
    def __init__(self,gamma,board, rewards, filename,p_success):                                       #instructor
        self.rows = board.shape[0]                       #rows
        self.columns = board.shape[1]                    #columns
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']        #ACTIONS
        self.legal_states = [] # 0 and 1                      #STATES as arrays
        self.terminal_states = []
        self.active_states = []
        self.gamma=gamma
        self.board = board      #
        self.rewards = rewards
        self.filename = filename
        self.p_success = p_success
        for row in range(self.rows):                          #fulfilling all the coordinates matrices
            for column in range(self.columns):
                if self.is_blocked(row,column):
                    continue
                self.legal_states.append((row, column))
                if abs(self.rewards[row, column]) > 0.1:
                    self.terminal_states.append((row, column))
                else:
                    self.active_states.append((row, column))

    def is_blocked(self,row,column):                          #returns True if there is a wall, boolean
        if row < 0 or row >= self.rows or column < 0 or column >= self.columns:
            return True
        if self.board[row, column] == 0:
            return True
        return False

    def get_transition_probabilities(self, state, action,p_success):  #gets a coordinate and a move
        row, col = state      #coordinate
        if state in self.terminal_states:
            return []
        directions = {                       #dictionary of the moves and their diagonals
            'UP': {'chosenWay': (-1, 0), 'diag': [(-1, -1), (-1, 1)]},
            'DOWN': {'chosenWay': (1, 0), 'diag': [(1, -1), (1, 1)]},
            'LEFT': {'chosenWay': (0, -1), 'diag': [(-1, -1), (1, -1)]},
            'RIGHT': {'chosenWay': (0, 1), 'diag': [(-1, 1), (1, 1)]}
        }
        coords = directions[action]
        d_row, d_col = coords['chosenWay']
        target_row, target_col = row + d_row, col + d_col    #the move that we want

        if self.is_blocked(target_row, target_col):          #checks if its blocked
            return []

        transitions = {}                                       #probabilities check with dict
        p_diag = (1-p_success)/2
        chosenWay_state = (target_row, target_col)

        for (dr_diag, dc_diag) in coords['diag']:                #checks the diagonals ,
            diag_row, diag_col = row + dr_diag, col + dc_diag    #if its blocked it adds the prob to the original move option
            diag_state = (diag_row, diag_col)
            if self.is_blocked(diag_row, diag_col):
                p_success += p_diag
            else:                                                #if not blocked
                if diag_state in transitions:
                    transitions[diag_state] += p_diag
                else:
                    transitions[diag_state] = p_diag

        if chosenWay_state in transitions:                 #transisions is our chart of probabilities for the moves available
            transitions[chosenWay_state] += p_success
        else:
            transitions[chosenWay_state] = p_success

        return list(transitions.items())                   #list of tuples from dict transitions to solve the equation

    def q_VALUE(self, state, action, current_values):      #our prestigious algorithm
        transitions = self.get_transition_probabilities(state, action,self.p_success)
        if not transitions:
            return -float('inf')
        q_VALUE = 0.0
        for next_state, prob in transitions:                #utility calculation
            next_r, next_c = next_state                     #split the coordinate of the next move to a row and a column
            reward = self.rewards[next_r, next_c]
            future_value = current_values[next_r, next_c]
            q_VALUE += prob * (reward + self.gamma * future_value)  #Belman equation
        return q_VALUE

    def value_iteration(self, epsilon=0.01, max_iterations=100):    #calculates in iterative way the utility for every situation,using the Belman equation
        U = np.zeros((self.rows, self.columns))
        if(self.gamma!=0):
            stop_con = epsilon * (1 - self.gamma) / self.gamma
        else:
            stop_con=1e-6
        for (r, c) in self.terminal_states:     #initialize
            U[r, c] = self.rewards[r, c]

        for i in range(max_iterations):
            prev_U = U.copy()
            delta = 0

            for state in self.active_states:   #only in value 1
                r, c = state
                max_val = -float('inf')
                for action in self.actions:
                    q_val = self.q_VALUE(state, action, prev_U)
                    if q_val > max_val:
                        max_val = q_val
                U[r, c] = max_val
                diff = abs(U[r, c] - prev_U[r, c])
                if diff > delta:
                    delta = diff
            print(f"iteration {i}: max delta = {delta:.5f}")

            if delta < stop_con:
                print(f"converged after {i} iterations")
                break
        return U,i                                                    #returns the value of each square(map of grades)

    def extract_policy(self, values):
        policy = np.empty((self.rows, self.columns), dtype=object)
        for state in self.active_states:
            r, c = state
            best_action = None
            max_q_VALUE = -float('inf')
            for action in self.actions:
                q_val = self.q_VALUE(state, action, values)
                if q_val > max_q_VALUE:
                    max_q_VALUE = q_val
                    best_action = action
            policy[r, c] = best_action

        return policy                                                #returns the optimal policy(pie*)

    def display_policy(self, policy):  # printing the policy matrice
        arrows = {'UP': '↑', 'DOWN': '↓', 'LEFT': '←', 'RIGHT': '→', None: ' '
                  }
        print("\nOptimal Policy:")
        print("-" * (self.columns * 4 + 1))

        for r in range(self.rows):
            row_str = "|"
            for c in range(self.columns):
                if (r, c) in self.terminal_states:
                    action_char = 'o'  # target
                elif self.is_blocked(r, c):
                    action_char = 'x'  # wall
                else:
                    action = policy[r, c]
                    action_char = arrows.get(action, ' ')
                row_str += f" {action_char} |"
            print(row_str)
            print("-" * (self.columns * 4 + 1))

    def save_value_image(self,values,iteration,algo_name):                  #saving the value matrice as a jpg in our folder
        limit = np.max(np.abs(values))
        if limit==0:limit=1  # avoids mistake if all the values are 0
        plt.imshow(values, cmap='seismic', vmin=-limit, vmax=limit)
        plt.colorbar()
        plt.title(f"{algo_name} Values - Ron Lapushner, Iterations: {iteration}")
        folder = 'Output'
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = f"{algo_name}_Values_RonLapushner.jpg"
        full_path = os.path.join(folder, filename)
        plt.savefig(full_path)
        plt.close()

    def save_policy_image(self, policy,algo_name):
        arrows = {'UP': '↑', 'DOWN': '↓', 'LEFT': '←', 'RIGHT': '→', None: ' '}
        table_data = []
        for r in range(self.rows):
            row_data = []
            for c in range(self.columns):
                if (r, c) in self.terminal_states:
                    action_char = 'o'
                elif self.is_blocked(r, c):
                    action_char = 'x'
                else:
                    action = policy[r, c]
                    action_char = arrows.get(action, ' ')
                row_data.append(action_char)
            table_data.append(row_data)

        plt.figure(figsize=(self.columns, self.rows))
        ax = plt.gca()
        ax.axis('off')
        table = plt.table(cellText=table_data, loc='center',cellLoc='center')
        plt.title(f"{algo_name} Policy - Ron Lapushner", fontsize=16)
        folder = 'Output'
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = f"{algo_name}_Policy_RonLapushner.jpg"
        full_path = os.path.join(folder, filename)
        plt.savefig(full_path, bbox_inches='tight')
        plt.close()

    def initialize_policy(self):
        policy=np.empty((self.rows, self.columns), dtype=object)
        priorities=['UP', 'DOWN', 'RIGHT', 'LEFT']                #the order from question number 2
        moves={'UP': (-1, 0), 'DOWN': (1, 0),'RIGHT': (0, 1),'LEFT': (0, -1)  } #dict

        for state in self.active_states:     #puts up if it legal
            r,c = state
            found=False
            for action in priorities:
                dr,dc=moves[action]
                targer_r,targer_c=r+dr,c+dc

                if not self.is_blocked(targer_r, targer_c):   #checks if its illegal move
                    policy[r,c]=action
                    found=True
                    break
            if not found:
                policy[r, c] = None
        return policy

    def policy_evaluation(self,policy,U,epsilon=0.01):
        stop_con = epsilon * (1 - self.gamma) / self.gamma     #our stop condition, similar to the VI algorithm
        steps=0
        for i in range(1000):
            steps+=1
            prev_U=U.copy()
            delta=0

            for state in self.active_states:
                r, c = state
                if state in self.terminal_states:
                    continue
                action = policy[r, c]
                if action is None:
                    continue
                new_val = self.q_VALUE(state, action, prev_U)
                U[r,c]=new_val
                diff = abs(U[r, c] - prev_U[r, c])
                if diff > delta:
                    delta = diff

            if delta<stop_con:
                break

        return U,steps

    def policy_iteration(self):
        pie=self.initialize_policy()    #using the function we made for initialization
        graph_data=[]
        iteration_index=0
        while True:
            iteration_index+=1
            U=np.zeros((self.rows, self.columns))
            for(r,c) in self.terminal_states:
                U[r,c]=self.rewards[r,c]
            U, eval_steps = self.policy_evaluation(pie, U)
            graph_data.append(eval_steps)
            flag=True

            for state in self.active_states:  #searching for the max move
                r,c=state
                current_action = pie[r, c]
                if current_action is None:
                    current_val=-float('inf')
                else:
                    current_val = self.q_VALUE(state, current_action, U)
                best_action = current_action
                max_val = current_val

                for action in self.actions:
                    val = self.q_VALUE(state, action, U)  #sending for the func q_value for calculation
                    if val > max_val + 0.000001:  #if we have found a max default it
                        max_val = val
                        best_action = action
                if best_action != current_action:
                    pie[r, c] = best_action
                    flag = False  #our policy is not the best, try again
            if flag:         #stop condition, if nothing has changed
                break
        return pie,U,iteration_index,graph_data

    def save_graph(self, evaluation_data):
        x_values = range(1, len(evaluation_data) + 1)
        y_values = evaluation_data
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, marker='o', linestyle='-', color='blue')
        plt.title("Policy Iteration Graph")
        plt.xlabel("Policy Iteration Number")
        plt.ylabel("Simplified Value Iteration")
        plt.grid(True)
        folder = 'Output'
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = "PolicyIteration_RonLapushner.jpg"
        full_path = os.path.join(folder, filename)
        plt.savefig(full_path)
        plt.close()


def print_usage():                 #explains how to run the program
    print("Usage: python MDP.py <input_file.npz> <ValueIteration|PolicyIteration>")


if __name__ == '__main__':
    if len(sys.argv) != 3 or sys.argv[2] not in ('ValueIteration', 'PolicyIteration'):   #analyzing the terminal order
        print_usage()
        sys.exit(1)
    filename = sys.argv[1]
    algorithm_type = sys.argv[2]
    data = np.load(filename)  # file loading
    board = data['states']  # states matrix called board
    rewards = data['rewards']  # reward matrix called rewards
    gamma=0.9
    p_success = 0.8                 #possibility for success
    mdp_agent = MDP(gamma, board, rewards, filename,p_success) #creating the instance
    if algorithm_type == 'ValueIteration':      #if the terminal order is: python MDP.py input1_2026a.npz ValueIteration
        U, iterations = mdp_agent.value_iteration()
        policy = mdp_agent.extract_policy(U)
        mdp_agent.save_value_image(U, iterations, "ValueIteration")
        mdp_agent.save_policy_image(policy, "ValueIteration")
        mdp_agent.display_policy(policy)
    else:                                       #if the terminal order is: python MDP.py input1_2026a.npz PolicyIteration
        policy, U, iterations, eval_history = mdp_agent.policy_iteration()
        mdp_agent.save_value_image(U, iterations, "PolicyIteration")
        mdp_agent.save_policy_image(policy, "PolicyIteration")
        mdp_agent.display_policy(policy)
        mdp_agent.save_graph(eval_history)
