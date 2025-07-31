import math
import random
from collections import defaultdict
from query_engine import QueryEngine

class MCTSNode:
    def __init__(self, state=None, parent=None, action=None):
        self.state = state or {}  # Partial mapping of tables to DB types
        self.parent = parent
        self.action = action      # (table, db_type) that led to this state
        self.children = []
        self.visits = 0
        self.value = 0
        self.untried_actions = None  # will be populated when expanded

    def is_fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0
    
    def is_terminal(self):
        # A node is terminal if all tables have been assigned
        return len(self.state) == len(ALL_TABLES)
    
    def expand(self, all_tables, db_options):
        if self.untried_actions is None:
            # Generate all possible actions from this state
            self.untried_actions = []
            for table in all_tables:
                if table not in self.state:
                    for db_type in db_options:
                        self.untried_actions.append((table, db_type))
                        
            print(f"Expanding node with state: {self.state}, untried actions: {self.untried_actions}")
            print("number of untried actions:", len(self.untried_actions))

        if not self.untried_actions:
            return None
            
        # Choose a random untried action
        action = random.choice(self.untried_actions)
        self.untried_actions.remove(action)
        
        # Create new state by applying the action
        table, db_type = action
        new_state = dict(self.state)
        new_state[table] = db_type
        
        # Create child node
        child = MCTSNode(state=new_state, parent=self, action=action)
        self.children.append(child)
        return child
    
    def best_child(self, exploration_weight=1.0):
        if not self.children:
            return None
            
        def ucb1(child):
            exploitation = child.value / child.visits if child.visits > 0 else 0
            exploration = math.sqrt(2 * math.log(self.visits) / child.visits) if child.visits > 0 else float('inf')
            return exploitation + exploration_weight * exploration
            
        return max(self.children, key=ucb1)

    def backpropagate(self, reward):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)


def test_mapping_performance(mapping):
    query_engine = QueryEngine(schema_mapping=mapping)
    execution_time = query_engine.run_queries(verbose=False)
    reward = 1.0 / (execution_time + 0.001) 
    return execution_time, reward


def complete_random_mapping(partial_mapping, all_tables, db_options):
    """Complete a partial mapping with random assignments"""
    complete_mapping = dict(partial_mapping)
    for table in all_tables:
        if table not in complete_mapping:
            complete_mapping[table] = random.choice(db_options)
    return complete_mapping


def mcts_search(all_tables, db_options, iterations=20, exploration_weight=1.0):
    print("Starting MCTS search...")
    initial_state = {'customer': 'mongodb', 'orders': 'neo4j', 'lineitem': 'mongodb'}
    root = MCTSNode(state=initial_state)
    best_mapping = None
    best_time = float('inf')
    
    for _ in range(iterations):
        print(f"Iteration {_ + 1}/{iterations}")
        
        
        # Selection
        node = root
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child(exploration_weight)
        
        # Expansion
        if not node.is_terminal():
            node = node.expand(all_tables, db_options)
            if node is None:  # No more actions to try
                continue
        
        # Simulation
        simulation_mapping = complete_random_mapping(node.state, all_tables, db_options)
        execution_time, reward = test_mapping_performance(simulation_mapping)
        
        if execution_time < best_time:
            best_time = execution_time
            best_mapping = simulation_mapping
            print(f"New best time: {best_time} with mapping: {best_mapping}")
        
        # Backpropagation
        node.backpropagate(reward)
    
    return best_mapping, best_time


def main():
    global ALL_TABLES
    ALL_TABLES = ['customer','orders', 'lineitem']
    db_options = ['postgres', 'mongodb', 'neo4j']
    
    best_mapping, best_time = mcts_search(ALL_TABLES, db_options, iterations=20)
    
    print(f"Final best time: {best_time} with mapping: {best_mapping}")



if __name__ == "__main__":
    main()