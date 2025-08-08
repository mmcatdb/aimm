import random
from query_engine import QueryEngine
import math
from typing import Dict, Tuple


class MCTSNode:
    def __init__(self, state, parent): 
        self.state = state                            # e.g. ('postgres', 'neo4j', 'postgres')
        self.children = []
        self.parents = [parent] if parent else []     #! Not actually used anywhere, so can remove it if no use case comes up  

        MCTS.explored_states.add(state)
        MCTS.state_to_node_map[state] = self
        self.value = 0
        self.visits = 0
        self.actions_left = self.get_base_actions()   # A list of pairs (table_id, database_name)

    def avg_value(self):
        return self.value/self.visits if self.visits > 0 else 0

    @staticmethod
    def get_updated_state(state, action):
        index, new_db = action
        new_state = list(state)
        new_state[index] = new_db
        return tuple(new_state)

    def get_base_actions(self):
        actions = []
        for i, current_db in enumerate(self.state):
            for db in MCTS.ALL_DBS:
                if db == current_db: continue

                action = (i, db)
                actions.append(action)

        return actions
    
    
    def is_reverting_action(self, action, previous_states):
        return MCTSNode.get_updated_state(self.state, action) in previous_states
    
    def get_viable_actions(self, previous_states):
        # Viable actions are actions that do not lead to any of the previous states
        return [action for action in self.actions_left if not self.is_reverting_action(action, previous_states)]

    def is_fully_expanded(self, previous_states):
        # Is fully expanded if there are no viable actions left.
        viable_actions = self.get_viable_actions(previous_states)
        return not viable_actions
    
    def is_leaf(self, path):
        # Is leaf if it has no children that aren't in the path already
        return not set(self.children) - set(path)

    def expand(self, previous_states):
        viable_actions = self.get_viable_actions(previous_states)
        if len(viable_actions) == 0: raise RuntimeError("ERROR: Trying to expand a node with no viable actions")
        
        action = random.choice(viable_actions)
        self.actions_left.remove(action)

        new_state = MCTSNode.get_updated_state(self.state, action)
        is_new_node = new_state not in MCTS.explored_states

        if is_new_node:
            node = MCTSNode(state=new_state, parent=self)
        else:
            node = MCTS.state_to_node_map[new_state]
            node.parents.append(self)

        self.children.append(node)

        return node, is_new_node
    

    def update_stats(self, reward):
        self.visits += 1
        self.value += reward

    def get_best_child(self, path, exploration_weight=1.414):
        viable_children = list(set(self.children) - set(path))
        if not viable_children:
            raise RuntimeError(f"Trying to go down a node with no viable children:\n  Node's children: {self.children}\n  Current path: {path}")
        

        #? Could work with edge statistics instead of node statistics to eliminate shared-visit distortion. 
        #   e.g. have a dictionary with (parent_id, child_id) keys and (value, visits) values
        #   ... Is that the desired behaviour though?


        total_visits_in_children = sum(child.visits for child in viable_children)
            
        def ucb1(child):
            exploitation = child.value / child.visits if child.visits > 0 else 0

            # Standard exploration formula:
            # exploration = math.sqrt(math.log(self.visits) / child.visits) if child.visits > 0 else float('inf')

            # Generalized formula for DAGs. For trees it has the same effect as the above formula
            exploration = math.sqrt(math.log(1 + total_visits_in_children) / child.visits) if child.visits > 0 else float('inf')

            return exploitation + exploration_weight*exploration
            
        # In case of ties, select randomly among the highest-scored children
        best_score = max(ucb1(child) for child in viable_children)
        best_children = [child for child in viable_children if ucb1(child) == best_score]
        return random.choice(best_children)



class MCTS:
    ALL_TABLES = ('customer', 'orders', 'lineitem')
    ALL_DBS = ('postgres', 'mongodb', 'neo4j')
    explored_states = set()
    state_to_node_map: Dict[Tuple[str,...], MCTSNode] = dict()


    @staticmethod
    def perform_simulation(mapping):
        query_engine = QueryEngine(schema_mapping=mapping)
        execution_time = query_engine.run_queries(verbose=False)
        reward = 100 / (execution_time + 0.001) 
        return reward, execution_time

    @staticmethod
    def state_to_mapping(state):
        return dict(zip(MCTS.ALL_TABLES, state))

    @staticmethod
    def run_mcts(cur_state, iterations=10):
        MCTS.explored_states.clear()
        MCTS.state_to_node_map.clear()

        best_mapping = MCTS.state_to_mapping(cur_state)
        _, exec_time = MCTS.perform_simulation(best_mapping)
        best_time = exec_time


        root = MCTSNode(state=cur_state, parent=None)

        for i in range(iterations):
            print(f'Iteration {i+1}')

            # TODO? Make path a set straight away, since there are several to-set conversions in MCTSNode methods
            path = [root]
            previous_states = {cur_state}   # Set for the O(1) membership checks
            node = root

            # 1. Selection
            while True:
                if not node.is_fully_expanded(previous_states) or node.is_leaf(path):
                    break
                node = node.get_best_child(path)
                path.append(node)
                previous_states.add(node.state)

            print(f"Selected node to expand with state: {node.state}")

            # 2. Expansion
            if not node.is_fully_expanded(previous_states):
                node, is_new_node = node.expand(previous_states)
                path.append(node)
                previous_states.add(node.state)

                if is_new_node:
                    # 3. Simulation
                    schema_mapping = MCTS.state_to_mapping(node.state) 
                    reward, exec_time = MCTS.perform_simulation(schema_mapping)

                    print(f"Time: {exec_time}s with state: {node.state}")

                    if exec_time < best_time:
                        best_time = exec_time
                        best_mapping = schema_mapping
                else:
                    reward = node.avg_value()
            else:
                reward = node.avg_value()

            path_has_duplicates = len(path) != len(set(path))
            if path_has_duplicates: 
                raise RuntimeError(f"ERROR: path has some duplicate states:\n  {[node.state for node in path]}")

            # 4. Backpropagation
            for node in path:
                node.update_stats(reward)

        return best_mapping, best_time
    


def main():
    best_mapping, best_time = MCTS.run_mcts(("postgres", "postgres", "postgres"))
    print(f"Final best mapping: {best_mapping} with time {best_time}s")


if __name__ == "__main__":
    main()