import random
from query_engine import QueryEngine
import math


class MCTSNode:
    def __init__(self, state, parent, previous_states): 
        self.state = state                              # e.g. ('postgres', 'neo4j', 'postgres')
        self.children = []
        self.parent = parent
        self.previous_states = previous_states or set() # A set of states. Set is used for O(1) membership check
        self.value = 0
        self.visits = 0
        self.actions_left = self.get_viable_actions()

    @staticmethod
    def get_updated_state(state, action):
        index, new_db = action
        new_state = list(state)
        new_state[index] = new_db
        return tuple(new_state)

    def get_viable_actions(self):
        actions = []
        for i, current_db in enumerate(self.state):
            for db in MCTS.ALL_DBS:
                if db == current_db: continue

                action = (i, db)
                new_state = MCTSNode.get_updated_state(self.state, action)

                is_reverting_action = new_state in self.previous_states
                if not is_reverting_action:
                    actions.append(action)

        return actions
    
    def is_fully_expanded(self):
        return len(self.actions_left) == 0

    def expand(self):
        if len(self.actions_left) == 0: raise RuntimeError("ERROR: Trying to expand node with no actions left")
        
        action = random.choice(self.actions_left)
        self.actions_left.remove(action)

        new_state = MCTSNode.get_updated_state(self.state, action)
        new_previous_states = self.previous_states | {self.state}
        new_node = MCTSNode(state=new_state, parent=self, previous_states=new_previous_states)

        self.children.append(new_node)

        return new_node

    def backpropagate(self, reward):
        self.visits += 1
        self.value += reward

        # TODO multiple parents
        if self.parent:
            self.parent.backpropagate(reward)

    def get_best_child(self, exploration_weight=1.414):
        if not self.children:
            return None
        
        total_visits_in_children = sum(child.visits for child in self.children)
            
        def ucb1(child):
            exploitation = child.value / child.visits if child.visits > 0 else 0

            # Standard exploration formula
            # exploration = math.sqrt(math.log(self.visits) / child.visits) if child.visits > 0 else float('inf')

            # Generalized formula for DAGs. For trees, it has the same effect as the above formula
            exploration = math.sqrt(math.log(1 + total_visits_in_children) / child.visits) if child.visits > 0 else float('inf')
            return exploitation + exploration_weight*exploration
            
        return max(self.children, key=ucb1)



class MCTS:
    ALL_TABLES = ('customer', 'orders', 'lineitem')
    ALL_DBS = ('postgres', 'mongodb', 'neo4j') 

    @staticmethod
    def perform_simulation(mapping):
        query_engine = QueryEngine(schema_mapping=mapping)
        execution_time = query_engine.run_queries(verbose=False)
        reward = 1.0 / (execution_time + 0.001) 
        return reward, execution_time

    @staticmethod
    def state_to_mapping(state):
        return {
            'customer': state[0],
            'orders': state[1],
            'lineitem': state[2]
        }

    @staticmethod
    def run_mcts(cur_state, iterations=20):
        # TODO initialize best time and best mapping with cur_state
        best_time = float("inf")
        best_mapping = MCTS.state_to_mapping(cur_state)


        root = MCTSNode(state=cur_state, parent=None, previous_states=None)

        for i in range(iterations):
            print(f'Iteration {i+1}')

            node = root

            # 1. Selection
            while node.is_fully_expanded():
                node = node.get_best_child()

            print(f"Selected node to expand with state: {node.state}")

            # 2. Expansion
            node = node.expand()

            # 3. Simulation
            schema_mapping = MCTS.state_to_mapping(node.state) 
            reward, exec_time = MCTS.perform_simulation(schema_mapping)

            print(f"Time: {exec_time}s with state: {node.state}")

            if exec_time < best_time:
                best_time = exec_time
                best_mapping = schema_mapping


            # 4. Backpropagation
            node.backpropagate(reward)


        return best_mapping, best_time
    


def main():
    best_mapping, best_time = MCTS.run_mcts(("postgres", "postgres", "postgres"))
    print(f"Final best mapping: {best_mapping} with time {best_time}s")


if __name__ == "__main__":
    main()