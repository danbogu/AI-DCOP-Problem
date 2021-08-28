import random
import itertools
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean


class Problem:
    def __init__(self, prob_of_constraint, prob_of_cost, num_of_agents=30, domain_size=10, max_iter=1000):

        # prob_of_constraint - p1.
        self.prob_of_constraint = prob_of_constraint
        # prob_of_cost - p2.
        self.prob_of_cost = prob_of_cost
        self.num_of_agents = num_of_agents
        self.outbox = {}
        self.domain = [x + 1 for x in range(domain_size)]
        self.agents_list = []
        self.max_iter = max_iter
        # Attributes of MGM-2 #
        self.mgm_suggestions_outbox = {}

    def create_agents(self):
        '''
        create set of agents for the problem
        '''
        for agentID in range(self.num_of_agents):
            new_agent = Agent(agentID, self.domain, self)
            self.agents_list.append(new_agent)


    def create_constraints(self):
        '''
        create set of constraints for the agents in the problem
        '''
        # Get all possible agents combinations
        all_combinations = list(itertools.combinations(self.agents_list, 2))
        for pair in all_combinations:
            # Draw if constraint exists
            is_cons = random.choices([0, 1], [1 - self.prob_of_constraint, self.prob_of_constraint])
            if not is_cons[0]:
                continue


            all_values = list(itertools.product(self.domain, self.domain))


            # Add to neighbors lists
            pair[0].neighbors.add(pair[1])
            pair[1].neighbors.add(pair[0])

            for values in all_values:
                # Draw if constraint exists with those values
                is_cons_values = random.choices([0, 1], [1 - self.prob_of_cost, self.prob_of_cost])
                if not is_cons_values[0]:
                    cost = 0
                else:
                    # If constraint exists, draw the cost
                    cost = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                # Add to constraints dictionaries
                # dict[neighbor ID, my value, his value] =  cost
                pair[0].constraints[(pair[1].ID, values[0], values[1])] = cost
                pair[1].constraints[(pair[0].ID, values[1], values[0])] = cost


    def distribute_messages(self):
        ''' Put messages (agent assignments) from outbox in the agents inbox'''
        # (destination, origin, value)
        for agentID, messages in self.outbox.items():
            self.agents_list[agentID].inbox = copy.deepcopy(messages)

        # Empty the outbox in the end
        self.outbox = {}

    def distribute_mgm_friend_suggestions(self):
        ''' Put friend suggestions (of neighbor agents) from suggestions outbox in the agents suggestions inbox(MGM-2)'''
        for agentID, this_suggestions_list in self.mgm_suggestions_outbox.items():
            self.agents_list[agentID].suggestions_list = this_suggestions_list

        self.mgm_suggestions_outbox = {}

    def reset_mgm_parameters(self):
        '''reset all agents parameters for MGM-2 problem'''
        for agent in self.agents_list:
            agent.reset_MGM_parameters()


class Agent:
    def __init__(self, ID, values_domain, problem):
        self.domain = values_domain
        self.ID = ID
        self.curr_assignment = random.choice(values_domain)
        self.curr_assignment_cost = 0
        self.neighbors = set()
        self.inbox = {}
        # Each agent holds one variable so the domain of the variable is the domain of the agent.
        self.constraints = {}
        self.problem = problem
        # Attributes of MGM-2 #
        self.suggestions_list = []
        self.R = None
        self.friend = None
        self.offer = None
        self.linked = None
        self.switch = True
        self.neighbors_R_dict = {}
        self.potential_assignment = None

    def run_iteration_dsa(self, p_change=0.7):
        '''By chosen probability, run DSA(C) iteration on an agent to change assignment for that probability'''
        if len(self.inbox) > 0:
            value, cost = self.compute_messages_dsa()
            change = random.choices([0, 1], [1-p_change, p_change])
            if change[0]:
                self.curr_assignment = value
                self.curr_assignment_cost = cost
            else:
                self.curr_assignment_cost = self.calculate_cost(self.curr_assignment)
        self.send_current_assignment()

    def compute_messages(self):
        '''compute self constraints cost for the iteration given neighbor assignments (MGM-2)
        and looks for it's next best assignment'''
        if len(self.inbox) == 0:
            return -1, -1
        else:
            possibilities = {}
            for val in self.domain:
                curr_cost = 0
                for neighbor in self.neighbors:
                    # (neighbor, my value, his value, cost)
                    curr_cost += self.constraints[neighbor.ID, val, self.inbox[neighbor.ID]]
                possibilities[val] = curr_cost

            best_val = min(possibilities, key=possibilities.get)
            cost = possibilities[best_val]

            return best_val, cost

    def compute_messages_dsa(self):
        '''compute self constraints cost for the iteration given neighbor assignments (DSA)
        and looks for it's next best assignment'''
        possibilities = {}
        for val in self.domain:
            curr_cost = self.calculate_cost(val)
            possibilities[val] = curr_cost

        best_cost = min(list(possibilities.values()))
        good_assignments = [k for k, v in possibilities.items() if v == best_cost]
        if self.curr_assignment in good_assignments and len(good_assignments) > 1:
            good_assignments.remove(self.curr_assignment)
        next_assignment = random.choice(good_assignments)

        return next_assignment, best_cost

    def send_current_assignment(self):
        '''create a current assignment massage and deliver it'''
        message = {self.ID: self.curr_assignment}
        for agent in self.neighbors:
            self.send_message(agent, message)

    def send_message(self, agent, message):
        '''Put self agent assignment in the mutual outbox corresponding to its neighbors'''
        if agent.ID in self.problem.outbox.keys():
            self.problem.outbox[agent.ID].update(copy.deepcopy(message))
        else:
            self.problem.outbox[agent.ID] = copy.deepcopy(message)  # initialize dict

    def calculate_cost(self, val=None):
        """Calculates the current personal cost for an agent's value"""
        if val is None:
            # If no value was specified, take the current assignment
            val = self.curr_assignment
        curr_cost = 0
        for neighbor in self.neighbors:
            # (neighbor, my value, his value, cost)
            curr_cost += self.constraints[neighbor.ID, val, self.inbox[neighbor.ID]]

        return curr_cost

    # MGM-2 related functions #
    def is_offering(self):
        '''
        boolean function making a random (50/50) choice to be an offering agent for the current iteration
        '''

        self.offer = random.choice([True, False])
        return self.offer

    def send_friend_suggestion(self):
        '''
        This function is built based on the assignment message distribution algorithm.
        When an agent is requested to send friend suggestion to a neighbor
        it adds or creates a pointer to itself in the outbox in the list - NOTE THIS DIFFERENCE HERE WE
        USE A LIST INSTEAD OF DICTIONARY related to that neighbor. Later the system will distribute this suggestion to
        the relevant neighbor.
        '''
        # If there are no neighbors, the agent can not choose friends
        if len(self.neighbors) == 0 or not self.offer:
            return

        # Pick a random neighbor:
        agent = random.choice(list(self.neighbors))
        # Put it in the outbox:
        if agent.ID in self.problem.mgm_suggestions_outbox.keys():
            self.problem.mgm_suggestions_outbox[agent.ID].append(self)
        else:
            self.problem.mgm_suggestions_outbox[agent.ID] = [self]  # initialize list
        # Note that if an agent is suggesting it will not activate 'choose_suggestion', hence it decline all suggestions


    def choose_suggestion(self):
        """Performs the match between 2 agents (MGM-2)"""
        # we choose only one "friend" from the list and then clear the list - hence, decline all other suggestions
        self.friend = random.choice(self.suggestions_list)
        self.friend.linked = self.ID  # We tell the friend that it is linked to a pair


    def compute_best_assignment(self):
        """Finds the best alternative assignment and its cost (MGM-2)"""
        possibilities = {}
        without_friend_neighbors = [neighbor for neighbor in list(self.neighbors) if neighbor.ID != self.friend.ID]
        without_linked_neighbors = [neighbor for neighbor in list(self.friend.neighbors) if neighbor.ID != self.ID]
        for my_val in self.domain:
            for friend_val in self.friend.domain:
                curr_cost = 0
                for neighbor in without_friend_neighbors:
                    # (neighbor, my value, his value, cost)
                    curr_cost += self.constraints[neighbor.ID, my_val, self.inbox[neighbor.ID]]
                for neighbor in without_linked_neighbors:
                    curr_cost += self.friend.constraints[neighbor.ID, friend_val, self.friend.inbox[neighbor.ID]]
                curr_cost += self.constraints[self.friend.ID, my_val, friend_val]
                possibilities[(my_val, friend_val)] = curr_cost

        best_cost = min(list(possibilities.values()))
        good_assignments = [k for k, v in possibilities.items() if v == best_cost]
        next_assignment = random.choice(good_assignments)

        return next_assignment, best_cost


    def compute_current_assignment_cost(self):
        """Calculates the personal cost of the current assignment (MGM-2)"""
        if self.friend is not None:
            # We are the computing agent:
            without_friend_neighbors = [neighbor for neighbor in list(self.neighbors) if neighbor.ID != self.friend.ID]
            curr_cost = 0
            for neighbor in without_friend_neighbors:
                # (neighbor, my value, his value, cost)
                curr_cost += self.constraints[neighbor.ID, self.curr_assignment, self.inbox[neighbor.ID]]
            for neighbor in self.friend.neighbors:
                curr_cost += self.friend.constraints[
                    neighbor.ID, self.friend.curr_assignment, self.friend.inbox[neighbor.ID]]
            return curr_cost

        else:
            # We are alone
            curr_cost = 0
            for neighbor in self.neighbors:
                curr_cost += self.constraints[neighbor.ID, self.curr_assignment, self.inbox[neighbor.ID]]
            return curr_cost

    def compute_R(self):
        """Calculates the best R the agent can get (MGM-2)
        and share it with agent neighbors
        """
        if len(self.neighbors) == 0:
            # No neighbors thus R is irrelevant
            return
        if self.linked is not None:
            # Not the computing agent
            return

        # If we are not linked - either we are alone or we are the computing agent
        curr_cost = self.compute_current_assignment_cost()

        if self.friend is not None:
            # We are the computing agent
            best_val, cost = self.compute_best_assignment()
            R_val = curr_cost - cost
            self.friend.R, self.R = R_val, R_val
            self.potential_assignment = best_val[0]
            self.friend.potential_assignment = best_val[1]
            #R sharing
            self.share_R_with_neighbors()
            self.friend.share_R_with_neighbors()

        else:
            # We are alone
            value, cost = self.compute_messages()
            self.R = curr_cost - cost
            self.potential_assignment = value
            # R sharing
            self.share_R_with_neighbors()


    def share_R_with_neighbors(self):
        """Send my R to all my non-partner neighbors"""
        if len(self.neighbors) == 0:
            # Agent is unconstrained
            return

        elif (self.linked is not None) or (self.friend is not None):  # I am in a couple
            # Share R with neighbors except my partner
            # Make a new neighbors list if I have a link or a friend
            try:
                without_friend_neighbors = [neighbor for neighbor in list(self.neighbors) if
                                            neighbor.ID != self.friend.ID]
                for agent in without_friend_neighbors:
                    agent.neighbors_R_dict[self.ID] = self.R
            except:
                without_friend_neighbors = [neighbor for neighbor in list(self.neighbors) if neighbor.ID != self.linked]
                for agent in without_friend_neighbors:
                    agent.neighbors_R_dict[self.ID] = self.R

        else:
            # Agent is not in a partnership
            for agent in self.neighbors:
                agent.neighbors_R_dict[self.ID] = self.R

    def compare_R(self):
        """Compares my R with that of my neighbors to determine who should change assignment"""
        if len(self.neighbors) == 0:
            # The agent is unconstrained, thus meaningless
            self.switch = False
            return

        if self.R <= 0:
            # R must be positive to be valid
            self.switch = False
            return

        if not self.neighbors_R_dict:
            # The dictionary is empty but we have neighbors, hence, our neighbor is our friend
            # Therefore, we want to switch to our best assignment combination with it
            return

        for neighbor in self.neighbors_R_dict:
            R_value = self.neighbors_R_dict[neighbor]
            if R_value > self.R:
                # My R is not the best in my area
                self.switch = False
            elif R_value == self.R:
                # Tie breaker by ID
                if neighbor > self.ID:
                    self.switch = False


    def change_assignment(self):
        """Checks if we have the winning R. If so, changes assignment. In the end the current assignment is sent."""
        if self.linked is not None:
            # I am only linked, I cannot decide.
            return
        elif self.friend is not None:
            # I am the computing agent, lets see what is going on with me and my friend.
            if self.switch and self.friend.switch:  # We are both good to go.
                self.curr_assignment = copy.deepcopy(self.potential_assignment)
                self.friend.curr_assignment = copy.deepcopy(self.friend.potential_assignment)
            self.send_current_assignment()
            self.friend.send_current_assignment()
            # self.friend.reset_MGM_parameters()
            # self.reset_MGM_parameters()
        else:
            # I am all alone with no partner this time, lets see if my R turned out to be the best.
            if self.switch:
                self.curr_assignment = copy.deepcopy(self.potential_assignment)
            self.send_current_assignment()
            # self.reset_MGM_parameters()


    def reset_MGM_parameters(self):
        '''
        self agent reset of all MGM-2 relared parameters before next iteration
        '''
        self.suggestions_list = []
        self.R = None
        self.friend = None
        self.offer = None
        self.linked = None
        self.switch = True
        self.neighbors_R_dict = {}
        self.potential_assignment = None


def total_prob_cost(prob):
    """Calculates the current global cost"""
    return sum([agent.calculate_cost() for agent in prob.agents_list])

def run_dsa(prob):
    """This is the main function operating the DSA-C algorithm.

    Args:
        prob (Problem): a DCOP to solve.

    Returns:
        a list of the achieved global cost in a 10 iterations gap.
    """
    results = []  # Will store the global cost
    for agent in prob.agents_list:
        # Send the initial (random) assignment
        agent.send_current_assignment()
    for i in range(1000):
        prob.distribute_messages()  # From previous iteration
        if i % 10 == 0:
            # Add to results every 10 iterations
            results.append(sum([a.calculate_cost() for a in prob.agents_list])/2)
        for agent in prob.agents_list:
            # The agent's iteration
            agent.run_iteration_dsa()

    prob.distribute_messages()
    results.append(sum([a.calculate_cost() for a in prob.agents_list])/2)
    return results

def run_mgm2(prob):
    """This is the main function operating the MGM-2 algorithm.

        Args:
            prob (Problem): a DCOP to solve.

        Returns:
            a list of the achieved global cost in a 10 iterations gap.
        """
    results = []
    # Initial message distribution iteration
    for agent in prob.agents_list:
        agent.send_current_assignment()

    for i in range(200):
        prob.distribute_messages()  # From previous iteration
        results.append(sum([a.calculate_cost() for a in prob.agents_list]) / 2)
        # Each agent chooses to offer and sends friend suggestions to it's neighbors
        for agent in prob.agents_list:
            if agent.is_offering():
                agent.send_friend_suggestion()
        # The agents receive it's friend suggestions
        prob.distribute_mgm_friend_suggestions()
        results.append(sum([a.calculate_cost() for a in prob.agents_list]) / 2)
        # Each agent chooses to respond
        for agent in prob.agents_list:
            if len(agent.suggestions_list) != 0 and not agent.offer:
                agent.choose_suggestion()
        results.append(sum([a.calculate_cost() for a in prob.agents_list]) / 2)
        for agent in prob.agents_list:
            agent.compute_R()
        results.append(sum([a.calculate_cost() for a in prob.agents_list]) / 2)
        for agent in prob.agents_list:
            agent.compare_R()
        results.append(sum([a.calculate_cost() for a in prob.agents_list]) / 2)
        for agent in prob.agents_list:
            agent.change_assignment()

        prob.reset_mgm_parameters()

    prob.distribute_messages()
    results.append(sum([a.calculate_cost() for a in prob.agents_list]) / 2)
    return results[0::10]


def run_algorithms(p1, p2):
    """Generates a random DCOP problem and runs DSA-C and MGM-2 on it

    Args:
        p1 (float): the probability for 2 agents to have constraints.
        P2 (float): the probability for a constraint of 2 values to be > 0.

    Returns:
        2 lists of results: for DSA and for MGM-2.
    """
    prob = Problem(p1, p2)
    prob.create_agents()
    prob.create_constraints()
    prob_dsa = copy.deepcopy(prob)
    res_dsa = run_dsa(prob_dsa)
    res_mgm = run_mgm2(prob)

    return res_dsa, res_mgm

# Charts creation #

def chart1(p1):
    dsa_results = []
    mgm_results = []
    x = np.linspace(0.1, 0.9, 9)
    for i in range(10):
        problems = [Problem(p1, p) for p in x]
        dsa_results.append([])
        mgm_results.append([])
        for p in problems:
            p.create_agents()
            p.create_constraints()
            p_dsa = copy.deepcopy(p)
            res = run_dsa(p_dsa)
            dsa_results[i].append(res[-1])
            res = run_mgm2(p)
            mgm_results[i].append(res[-1])

    dsa_y = [mean(col) for col in zip(*dsa_results)]
    mgm_y = [mean(col) for col in zip(*mgm_results)]

    return x, dsa_y, mgm_y

def chart2(p1):
    dsa_results = []
    mgm_results = []
    x = list(range(0, 1001, 10))
    problems = [Problem(p1, 1) for i in range(10)]
    for p in problems:
        p.create_agents()
        p.create_constraints()
        p_dsa = copy.deepcopy(p)
        res = run_dsa(p_dsa)
        dsa_results.append(res)
        res = run_mgm2(p)
        mgm_results.append(res)

    dsa_y = [mean(col) for col in zip(*dsa_results)]
    mgm_y = [mean(col) for col in zip(*mgm_results)]

    return x, dsa_y, mgm_y


if __name__ == "__main__":
    res_dsa, res_mgm = run_algorithms(0.2, 0.5)
    print("DSA Results: " + str(res_dsa))
    print("MGM-2 Results: " + str(res_mgm))

