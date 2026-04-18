import math
import random
from typing import Protocol
from core.drivers import DriverType

QueryState = list[str]
StateMatrix = tuple[tuple[int, ...], ...]
StateMapping = list[str]
"""Assigned database for each query."""
StateKey = StateMatrix # Might be changed later
MCTSEdge = tuple[StateKey, StateKey]
MCTSAction = tuple[str, int, int, None]  # (mode, query_index, db_index, extra)

class IQueryEngine(Protocol):
    def estimate_latency(self, state: StateMapping) -> float: ...

class IOutputCollector(Protocol):
    def on_initial_solution(self, state: StateMapping, latency: float): ...

    def on_best_solution(self, state: StateMapping, latency: float, processed_states: int): ...

    def on_iteration(self, iteration: int, processed_states: int): ...

class MCTS:
    """
        Graph-based MCTS for matrix states.

    State format used by this implementation:
    - Immutable matrix (tuple of tuples).
        - Rows = databases, columns = queries.
    - Cell values:
            - False: query is not assigned to this database.
            - True: query is assigned to this database.
    """

    def __init__(
        self,
        query_engine: IQueryEngine,
        kinds: list[int],
        output_collector: IOutputCollector,
        databases: list[str],
        relation_endpoints=None,
        relationship_kinds=None,
        isa_roots=None,
        isa_specializations=None,
        allowed_isa_strategies=None,
        embedding_morphism_exists=None,
        exploration_weight=None,
    ):
        self.query_engine = query_engine
        self.kinds = tuple(kinds)
        self.query_ids = self.kinds
        self.output_collector = output_collector

        # if databases is None:
        #     self.databases = tuple(self.query_engine.available_databases())
        # else:
        #     self.databases = tuple(databases)

        self.databases = tuple(databases)

        # if not self.databases:
        #     raise ValueError('No databases available for MCTS')
        # if not self.kinds:
        #     raise ValueError('No queries provided for MCTS')

        self.kind_to_index = {kind: index for index, kind in enumerate(self.kinds)}
        self.db_to_index = {db: index for index, db in enumerate(self.databases)}

        # Optional schema metadata/hooks used by validation rules.
        self.relation_endpoints = relation_endpoints or {}
        self.relationship_kinds = set(relationship_kinds or [])
        self.isa_roots = set(isa_roots or [])
        self.isa_specializations = isa_specializations or {}
        self.allowed_isa_strategies = allowed_isa_strategies or {}
        self.embedding_morphism_exists = embedding_morphism_exists

        self.exploration_weight = math.sqrt(2.0) if exploration_weight is None else exploration_weight

        # Global graph storage.
        self.state_to_node = dict[StateKey, MCTSNode]()
        self.edge_visits = dict[MCTSEdge, int]()

    def run(self, initial_state: QueryState, iterations: int):
        if iterations <= 0:
            raise ValueError('iterations must be > 0')

        self.state_to_node.clear()
        self.edge_visits.clear()

        root_state = self.normalize_state(initial_state)
        if not self.is_valid_state(root_state):
            raise ValueError('Initial state is not valid according to current constraints')

        root, _ = self.get_or_create_node(root_state)

        processed_states = 0
        # Baseline evaluation (spec: measure root first).
        best_mapping = self._state_to_mapping(root_state)
        _, best_time = self.perform_simulation(root_state)

        self.output_collector.on_initial_solution(best_mapping, best_time)

        for iteration in range(iterations):
            self.output_collector.on_iteration(iteration, processed_states)

            path_nodes = [root]
            path_edges = []
            path_state_keys = {root.key}
            node = root

            # 1) Selection: descend while fully expanded and there is at least one viable child.
            while node.is_fully_expanded(path_state_keys) and not node.is_leaf(path_state_keys):
                child = node.select_best_child(path_state_keys)
                edge = (node.key, child.key)

                path_edges.append(edge)
                path_nodes.append(child)
                path_state_keys.add(child.key)
                node = child

            # 2) Expansion: add one action if possible.
            if not node.is_fully_expanded(path_state_keys):
                child, is_new = node.expand(path_state_keys)
                edge = (node.key, child.key)

                path_edges.append(edge)
                path_nodes.append(child)
                path_state_keys.add(child.key)

                if is_new:
                    processed_states += 1
                    reward, exec_time = self.perform_simulation(child.state)
                    if exec_time < best_time:
                        self.output_collector.on_best_solution(self._state_to_mapping(child.state), exec_time, processed_states)

                        # print([schema, exec_time])
                        best_time = exec_time
                        best_mapping = self._state_to_mapping(child.state)
                else:
                    reward = child.avg_reward()
            else:
                reward = node.avg_reward()

            # 3) Backpropagation for nodes and traversed edges.
            for visited_node in path_nodes:
                visited_node.update(reward)

            for edge in path_edges:
                self.edge_visits[edge] = self.edge_visits.get(edge, 0) + 1

        return best_mapping, best_time

    def get_or_create_node(self, state: StateMatrix):
        key = self.state_key(state)
        node = self.state_to_node.get(key)
        if node is not None:
            return node, False

        node = MCTSNode(self, state)
        self.state_to_node[key] = node
        return node, True

    def state_key(self, state: StateMatrix) -> StateKey:
        # State is already immutable, but keep this helper for readability and future changes.
        return state

    def normalize_state(self, raw_state: QueryState) -> StateMatrix:
        """
        Accepts either:
        - Mapping style from old implementation: tuple/list of db names per kind column.
        - Matrix style: rows x columns with bool/string cells.

        Returns immutable matrix: tuple(tuple(...), ...)
        """

        # Backward-compatible input: (db_for_query0, db_for_query1, ...)
        if self.looks_like_mapping_vector(raw_state):
            return self.mapping_vector_to_state(raw_state)

        # Matrix input.
        matrix = []
        if not isinstance(raw_state, (list, tuple)):
            raise ValueError('State must be a mapping vector or matrix')

        if len(raw_state) != len(self.databases):
            raise ValueError('Matrix row count must match number of databases')

        for row_index, row in enumerate(raw_state):
            if not isinstance(row, (list, tuple)):
                raise ValueError('Each matrix row must be list/tuple')
            if len(row) != len(self.kinds):
                raise ValueError('Matrix column count must match number of queries')

            normalized_row = []
            for column_index, cell in enumerate(row):
                normalized_row.append(self.normalize_cell(cell, row_index, column_index))
            matrix.append(tuple(normalized_row))

        return tuple(matrix)

    def looks_like_mapping_vector(self, raw_state: QueryState) -> bool:
        if not isinstance(raw_state, (list, tuple)):
            return False
        if len(raw_state) != len(self.kinds):
            return False

        for value in raw_state:
            if value not in self.db_to_index:
                return False
        return True

    def mapping_vector_to_state(self, mapping_vector: QueryState) -> StateMatrix:
        # Start all cells as False.
        rows = []
        for _ in self.databases:
            rows.append([False] * len(self.kinds))

        for query_index, db_name in enumerate(mapping_vector):
            db_index = self.db_to_index[db_name]
            rows[db_index][query_index] = True

        return tuple(tuple(row) for row in rows)

    # def state_to_mapping_vector(self, state: StateMatrix) -> QueryState:
    #     mapping_vector = []
    #     for query_index in range(len(self.kinds)):
    #         assigned_db = None
    #         for db_index in range(len(self.databases)):
    #             cell = state[db_index][query_index]
    #             if cell is True:
    #                 assigned_db = self.databases[db_index]
    #                 break

    #         if assigned_db is None:
    #             raise ValueError('Invalid state: query has no assigned database')
    #         mapping_vector.append(assigned_db)

    #     return tuple(mapping_vector)

    def normalize_cell(self, cell, row_index, column_index) -> bool:
        if cell is False or cell is True:
            return cell

        if cell is None:
            return False

        if isinstance(cell, str):
            lowered = cell.strip().lower()
            if lowered == 'false':
                return False
            if lowered == 'true':
                return True

        raise ValueError('Invalid cell value in matrix state')

    def perform_simulation(self, state):
        """
        Evaluates the state and returns (reward, execution_time).

        Tries multiple integration styles:
        1) query_engine.estimate_latency(state_matrix, mcts)
        2) query_engine.measure_state(state_matrix)
        3) query_engine.measure_queries(mapping)
        """

        # if hasattr(self.query_engine, 'estimate_latency'):
        #     execution_time = self.query_engine.estimate_latency(state, self)
        # elif hasattr(self.query_engine, 'measure_state'):
        #     execution_time = self.query_engine.measure_state(state)
        # elif hasattr(self.query_engine, 'measure_queries'):
        #     execution_time = self.query_engine.measure_queries(self._state_to_mapping(state), verbose=False)
        # else:
        #     raise AttributeError('query_engine must provide estimate_latency, measure_state, or measure_queries')

        execution_time = self.query_engine.estimate_latency(self._state_to_mapping(state))

        reward = 100.0 / (execution_time + 0.001)
        return reward, execution_time

    def _state_to_mapping(self, state: StateMatrix) -> StateMapping:
        """
        Converts matrix state into query -> database mapping.
        """

        mapping = list[str]()
        for query_index, query_id in enumerate(self.kinds):
            chosen_db = None
            for db_index, db_name in enumerate(self.databases):
                cell = state[db_index][query_index]
                if cell is True:
                    chosen_db = db_name
                    break

            if chosen_db is None:
                raise ValueError(f'Query has no assigned database in state: {query_id}')

            mapping.append(chosen_db)

        return mapping

    def is_valid_state(self, state: StateMatrix) -> bool:
        return self.validate_query_assignments(state)

    def validate_query_assignments(self, state: StateMatrix) -> bool:
        for query_index in range(len(self.kinds)):
            assignment_count = 0
            for db_index in range(len(self.databases)):
                cell = state[db_index][query_index]
                if not isinstance(cell, bool):
                    return False
                if cell:
                    assignment_count += 1
            if assignment_count != 1:
                return False
        return True

    def validate_structural_integrity(self, state: StateMatrix) -> bool:
        for kind_index, kind_name in enumerate(self.kinds):
            if kind_name in self.relationship_kinds:
                continue
            if kind_name in self.isa_roots:
                continue

            has_storage = False
            for db_index in range(len(self.databases)):
                if state[db_index][kind_index] is not False:
                    has_storage = True
                    break

            if not has_storage:
                return False

        return True

    def validate_embeddings(self, state: StateMatrix) -> bool:
        mongo_index = None
        for index, db_name in enumerate(self.databases):
            if isinstance(db_name, str) and db_name == DriverType.MONGO.value:
                mongo_index = index
                break

        for db_index in range(len(self.databases)):
            graph = dict[int, int]()

            # Build local embedding graph for this database.
            for kind_index, kind_name in enumerate(self.kinds):
                cell = state[db_index][kind_index]
                if isinstance(cell, str):
                    # Temporary product rule: embeddings can only exist in MongoDB.
                    if mongo_index is None or db_index != mongo_index:
                        return False

                    # Embedded target must exist in the same database.
                    target_kind_index = self.kind_to_index[cell]
                    if state[db_index][target_kind_index] is False:
                        return False

                    if self.embedding_morphism_exists is not None:
                        if not self.embedding_morphism_exists(cell, kind_name):
                            return False
                    graph[kind_name] = cell

            # Detect cycles in embedding graph.
            if self.embedding_graph_has_cycle(graph):
                return False

        return True

    def embedding_graph_has_cycle(self, graph: dict[int, int]) -> bool:
        visiting = set()
        visited = set()

        def dfs(node: int) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False

            visiting.add(node)
            target = graph.get(node)
            if target is not None and dfs(target):
                return True
            visiting.remove(node)
            visited.add(node)
            return False

        for node_name in graph:
            if dfs(node_name):
                return True
        return False

    def validate_relationship_kinds(self, state: StateMatrix) -> bool:
        if not self.relationship_kinds:
            return True

        for rel_kind in self.relationship_kinds:
            if rel_kind not in self.kind_to_index:
                continue

            rel_kind_index = self.kind_to_index[rel_kind]

            # Cannot be embedded: must be True/False only.
            for db_index in range(len(self.databases)):
                cell = state[db_index][rel_kind_index]
                if isinstance(cell, str):
                    return False

            # Optional endpoint constraints if endpoints metadata exists.
            endpoints = self.relation_endpoints.get(rel_kind)
            if not endpoints:
                continue

            if len(endpoints) != 2:
                continue

            left_kind, right_kind = endpoints
            if left_kind not in self.kind_to_index or right_kind not in self.kind_to_index:
                continue

            left_index = self.kind_to_index[left_kind]
            right_index = self.kind_to_index[right_kind]

            left_db = self.kind_primary_db(state, left_index)
            right_db = self.kind_primary_db(state, right_index)

            if left_db is None or right_db is None:
                return False
            if left_db != right_db:
                return False

            # If relationship is stored, it must live in that same database.
            relation_db = self.kind_primary_db(state, rel_kind_index)
            if relation_db is not None and relation_db != left_db:
                return False

            # Cannot connect objects embedded into each other / same common kind.
            if self.are_embedded_into_each_other(state, left_index, right_index):
                return False
            if self.are_embedded_into_same_host(state, left_index, right_index):
                return False

        return True

    def validate_isa_rules(self, state: StateMatrix) -> bool:
        if not self.isa_roots:
            return True

        for root_kind in self.isa_roots:
            if root_kind not in self.kind_to_index:
                continue

            root_index = self.kind_to_index[root_kind]
            specializations = self.isa_specializations.get(root_kind, [])

            for db_index, db_name in enumerate(self.databases):
                strategies = self.allowed_isa_strategies.get((db_name, root_kind), {'single table', 'table per class'})
                root_cell = state[db_index][root_index]

                if 'single table' not in strategies and root_cell is not False:
                    return False

                if 'table per class' not in strategies:
                    for child_kind in specializations:
                        if child_kind not in self.kind_to_index:
                            continue
                        child_index = self.kind_to_index[child_kind]
                        if state[db_index][child_index] is not False:
                            return False

                # Avoid redundant root + specialization direct storage in same DB.
                if root_cell is True:
                    for child_kind in specializations:
                        if child_kind not in self.kind_to_index:
                            continue
                        child_index = self.kind_to_index[child_kind]
                        if state[db_index][child_index] is True:
                            return False

        return True

    def kind_primary_db(self, state: StateMatrix, kind_index: int) -> str | None:
        for db_index in range(len(self.databases)):
            if state[db_index][kind_index] is not False:
                return self.databases[db_index]
        return None

    def are_embedded_into_each_other(self, state: StateMatrix, left_index: int, right_index: int) -> bool:
        left_kind = self.kinds[left_index]
        right_kind = self.kinds[right_index]

        for db_index in range(len(self.databases)):
            left_cell = state[db_index][left_index]
            right_cell = state[db_index][right_index]
            if left_cell == right_kind and right_cell == left_kind:
                return True

        return False

    def are_embedded_into_same_host(self, state: StateMatrix, left_index: int, right_index: int) -> bool:
        for db_index in range(len(self.databases)):
            left_cell = state[db_index][left_index]
            right_cell = state[db_index][right_index]

            if isinstance(left_cell, str) and isinstance(right_cell, str) and left_cell == right_cell:
                return True

        return False


class MCTSNode:
    def __init__(self, mcts: MCTS, state: StateMatrix):
        self.mcts = mcts
        self.state = state
        self.key = mcts.state_key(state)

        self.children = list[MCTSNode]()
        self.parents = set()

        self.total_reward = 0.0
        self.visits = 0

        # For deterministic coverage of action space per node.
        self.actions_left = self.generate_base_actions()

    def avg_reward(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    def update(self, reward: float):
        self.visits += 1
        self.total_reward += reward

    def generate_base_actions(self) -> list[MCTSAction]:
        """
        Each action changes one query assignment.

        Action tuple format:
        ('direct', query_index, db_index, None)
        """

        actions = list[MCTSAction]()
        for query_index, _ in enumerate(self.mcts.kinds):
            for db_index in range(len(self.mcts.databases)):
                if query_index >= 9 and db_index == 2: continue  # NOTE: Mongo doesn't have last 2 queries defined
                actions.append(('direct', query_index, db_index, None))

        random.shuffle(actions)
        return actions

    def apply_action(self, action: MCTSAction) -> StateMatrix:
        mode, query_index, db_index, _ = action

        rows = [list(row) for row in self.state]

        # Reset this query column across all databases first (single assignment policy).
        for row_index in range(len(rows)):
            rows[row_index][query_index] = False

        if mode == 'direct':
            rows[db_index][query_index] = True
        else:
            raise ValueError('Unknown action mode')

        return tuple(tuple(row) for row in rows)

    def get_viable_actions(self, path_state_keys: set[StateKey]) -> list[MCTSAction]:
        viable = []
        for action in self.actions_left:
            new_state = self.apply_action(action)
            new_key = self.mcts.state_key(new_state)

            # Cycle prevention for current traversal path.
            if new_key in path_state_keys:
                continue

            if not self.mcts.is_valid_state(new_state):
                continue

            viable.append(action)

        return viable

    def is_fully_expanded(self, path_state_keys: set[StateKey]) -> bool:
        return len(self.get_viable_actions(path_state_keys)) == 0

    def is_leaf(self, path_state_keys: set[StateKey]) -> bool:
        for child in self.children:
            if child.key not in path_state_keys:
                return False
        return True

    def expand(self, path_state_keys: set[StateKey]):
        viable_actions = self.get_viable_actions(path_state_keys)
        if not viable_actions:
            raise RuntimeError('expand called on node without viable actions')

        action = random.choice(viable_actions)
        self.actions_left.remove(action)

        new_state = self.apply_action(action)
        child, is_new = self.mcts.get_or_create_node(new_state)

        # Link parent/child graph edges.
        if child not in self.children:
            self.children.append(child)
        child.parents.add(self)

        return child, is_new

    def select_best_child(self, path_state_keys: set[StateKey]) -> 'MCTSNode':
        viable_children = [child for child in self.children if child.key not in path_state_keys]
        if not viable_children:
            raise RuntimeError('select_best_child called with no viable child')

        best_score = None
        best_candidates = list[MCTSNode]()

        for child in viable_children:
            edge_key = (self.key, child.key)
            edge_visits = self.mcts.edge_visits.get(edge_key, 0)

            # UCT2 = avg(child) + C * sqrt(ln(parent_visits) / edge_visits)
            if edge_visits == 0:
                score = float('inf')
            else:
                exploitation = child.avg_reward()
                parent_visits = max(1, self.visits)
                exploration = math.sqrt(math.log(parent_visits) / edge_visits)
                score = exploitation + self.mcts.exploration_weight * exploration

            if best_score is None or score > best_score:
                best_score = score
                best_candidates = [child]
            elif score == best_score:
                best_candidates.append(child)

        return random.choice(best_candidates)
