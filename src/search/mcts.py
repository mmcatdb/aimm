import math
import random


class MCTS:
    """
    Graph-based MCTS for matrix states.

    State format used by this implementation:
    - Immutable matrix (tuple of tuples).
    - Rows = databases, columns = kinds.
    - Cell values:
      - False: kind is not stored in this database.
      - True: kind is stored directly in this database.
      - <kind_name>: kind is embedded into <kind_name> in this database.
    """

    def __init__(
        self,
        query_engine,
        kinds,
        databases=None,
        relation_endpoints=None,
        relationship_kinds=None,
        isa_roots=None,
        isa_specializations=None,
        allowed_isa_strategies=None,
        embedding_morphism_exists=None,
        custom_validators=None,
        exploration_weight=None,
        verbose=False,
    ):
        if query_engine is None:
            raise ValueError("query_engine must be provided")

        self.query_engine = query_engine
        self.kinds = tuple(kinds)

        if databases is None:
            self.databases = tuple(self.query_engine.available_databases())
        else:
            self.databases = tuple(databases)

        if not self.databases:
            raise ValueError("No databases available for MCTS")
        if not self.kinds:
            raise ValueError("No kinds provided for MCTS")

        self.kind_to_index = {kind: index for index, kind in enumerate(self.kinds)}
        self.db_to_index = {db: index for index, db in enumerate(self.databases)}

        # Optional schema metadata/hooks used by validation rules.
        self.relation_endpoints = relation_endpoints or {}
        self.relationship_kinds = set(relationship_kinds or [])
        self.isa_roots = set(isa_roots or [])
        self.isa_specializations = isa_specializations or {}
        self.allowed_isa_strategies = allowed_isa_strategies or {}
        self.embedding_morphism_exists = embedding_morphism_exists
        self.custom_validators = list(custom_validators or [])

        self.exploration_weight = math.sqrt(2.0) if exploration_weight is None else exploration_weight
        self.verbose = verbose

        # Global graph storage.
        self.state_to_node = {}
        self.edge_visits = {}

    def run(self, initial_state, iterations=100):
        if iterations <= 0:
            raise ValueError("iterations must be > 0")

        self.state_to_node.clear()
        self.edge_visits.clear()

        root_state = self.normalize_state(initial_state)
        if not self.is_valid_state(root_state):
            raise ValueError("Initial state is not valid according to current constraints")

        root, _ = self.get_or_create_node(root_state)

        # Baseline evaluation (spec: measure root first).
        best_mapping = self.state_to_mapping(root_state)
        _, best_time = self.perform_simulation(root_state)

        for iteration in range(iterations):
            if self.verbose:
                print("Iteration", iteration + 1)

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
                    reward, exec_time = self.perform_simulation(child.state)
                    if exec_time < best_time:
                        best_time = exec_time
                        best_mapping = self.state_to_mapping(child.state)
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

    def get_or_create_node(self, state):
        key = self.state_key(state)
        node = self.state_to_node.get(key)
        if node is not None:
            return node, False

        node = MCTSNode(self, state)
        self.state_to_node[key] = node
        return node, True

    def state_key(self, state):
        # State is already immutable, but keep this helper for readability and future changes.
        return state

    def normalize_state(self, raw_state):
        """
        Accepts either:
        - Mapping style from old implementation: tuple/list of db names per kind column.
        - Matrix style: rows x columns with bool/string cells.

        Returns immutable matrix: tuple(tuple(...), ...)
        """

        # Backward-compatible input: (db_for_kind0, db_for_kind1, ...)
        if self.looks_like_mapping_vector(raw_state):
            return self.mapping_vector_to_state(raw_state)

        # Matrix input.
        matrix = []
        if not isinstance(raw_state, (list, tuple)):
            raise ValueError("State must be a mapping vector or matrix")

        if len(raw_state) != len(self.databases):
            raise ValueError("Matrix row count must match number of databases")

        for row_index, row in enumerate(raw_state):
            if not isinstance(row, (list, tuple)):
                raise ValueError("Each matrix row must be list/tuple")
            if len(row) != len(self.kinds):
                raise ValueError("Matrix column count must match number of kinds")

            normalized_row = []
            for column_index, cell in enumerate(row):
                normalized_row.append(self.normalize_cell(cell, row_index, column_index))
            matrix.append(tuple(normalized_row))

        return tuple(matrix)

    def looks_like_mapping_vector(self, raw_state):
        if not isinstance(raw_state, (list, tuple)):
            return False
        if len(raw_state) != len(self.kinds):
            return False

        for value in raw_state:
            if value not in self.db_to_index:
                return False
        return True

    def mapping_vector_to_state(self, mapping_vector):
        # Start all cells as False.
        rows = []
        for _ in self.databases:
            rows.append([False] * len(self.kinds))

        for kind_index, db_name in enumerate(mapping_vector):
            db_index = self.db_to_index[db_name]
            rows[db_index][kind_index] = True

        return tuple(tuple(row) for row in rows)
    
    def state_to_mapping_vector(self, state):
        mapping_vector = []
        for kind_index in range(len(self.kinds)):
            assigned_db = None
            for db_index in range(len(self.databases)):
                cell = state[db_index][kind_index]
                if cell is not False:
                    assigned_db = self.databases[db_index]
                    break

            if assigned_db is None:
                raise ValueError("Invalid state: kind has no assigned database")
            mapping_vector.append(assigned_db)

        return tuple(mapping_vector)

    def normalize_cell(self, cell, row_index, column_index):
        if cell is False or cell is True:
            return cell

        if cell is None:
            return False

        if isinstance(cell, str):
            lowered = cell.strip().lower()
            if lowered == "false":
                return False
            if lowered == "true":
                return True

            # String identifier: must reference a known kind and not itself.
            kind_name = cell.strip()
            current_kind = self.kinds[column_index]
            if kind_name == current_kind:
                raise ValueError("Embedding target cannot be the same kind")
            if kind_name not in self.kind_to_index:
                raise ValueError("Unknown embedding identifier: " + kind_name)
            return kind_name

        raise ValueError("Invalid cell value in matrix state")

    def perform_simulation(self, state):
        """
        Evaluates the state and returns (reward, execution_time).

        Tries multiple integration styles:
        1) query_engine.estimate_latency(state_matrix)
        2) query_engine.measure_state(state_matrix)
        3) query_engine.measure_queries(mapping)
        """


        execution_time = self.query_engine.estimate_latency(state, self)

        reward = 100.0 / (execution_time + 0.001)
        return reward, execution_time

    def state_to_mapping(self, state):
        """
        Converts matrix state into kind -> database mapping.
        If a kind has multiple non-false placements (should not happen with default actions),
        this picks the first row where the kind appears.
        """

        mapping = {}
        for kind_index, kind_name in enumerate(self.kinds):
            chosen_db = None
            for db_index, db_name in enumerate(self.databases):
                cell = state[db_index][kind_index]
                if cell is not False:
                    chosen_db = db_name
                    break

            if chosen_db is None:
                # Keep behavior explicit so invalid states are obvious during integration.
                raise ValueError("Kind has no assigned database in state: " + kind_name)

            mapping[kind_name] = chosen_db

        return mapping

    def is_valid_state(self, state):
        # 3.1 Structural integrity: non-relationship, non-ISA kinds must appear at least once.
        if not self.validate_structural_integrity(state):
            return False

        # 3.2 Embedding rules.
        if not self.validate_embeddings(state):
            return False

        # 3.3 Relationship kind rules.
        if not self.validate_relationship_kinds(state):
            return False

        # 3.4 ISA rules.
        if not self.validate_isa_rules(state):
            return False

        # Additional project-specific rules.
        for validator in self.custom_validators:
            if not validator(state, self):
                return False

        return True

    def validate_structural_integrity(self, state):
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

    def validate_embeddings(self, state):
        mongodb_index = None
        for index, db_name in enumerate(self.databases):
            if isinstance(db_name, str) and db_name.lower() == "mongodb":
                mongodb_index = index
                break

        for db_index in range(len(self.databases)):
            graph = {}

            # Build local embedding graph for this database.
            for kind_index, kind_name in enumerate(self.kinds):
                cell = state[db_index][kind_index]
                if isinstance(cell, str):
                    # Temporary product rule: embeddings can only exist in MongoDB.
                    if mongodb_index is None or db_index != mongodb_index:
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

    def embedding_graph_has_cycle(self, graph):
        visiting = set()
        visited = set()

        def dfs(node):
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

    def validate_relationship_kinds(self, state):
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

    def validate_isa_rules(self, state):
        if not self.isa_roots:
            return True

        for root_kind in self.isa_roots:
            if root_kind not in self.kind_to_index:
                continue

            root_index = self.kind_to_index[root_kind]
            specializations = self.isa_specializations.get(root_kind, [])

            for db_index, db_name in enumerate(self.databases):
                strategies = self.allowed_isa_strategies.get((db_name, root_kind), {"single table", "table per class"})
                root_cell = state[db_index][root_index]

                if "single table" not in strategies and root_cell is not False:
                    return False

                if "table per class" not in strategies:
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

    def kind_primary_db(self, state, kind_index):
        for db_index in range(len(self.databases)):
            if state[db_index][kind_index] is not False:
                return self.databases[db_index]
        return None

    def are_embedded_into_each_other(self, state, left_index, right_index):
        left_kind = self.kinds[left_index]
        right_kind = self.kinds[right_index]

        for db_index in range(len(self.databases)):
            left_cell = state[db_index][left_index]
            right_cell = state[db_index][right_index]
            if left_cell == right_kind and right_cell == left_kind:
                return True

        return False

    def are_embedded_into_same_host(self, state, left_index, right_index):
        for db_index in range(len(self.databases)):
            left_cell = state[db_index][left_index]
            right_cell = state[db_index][right_index]

            if isinstance(left_cell, str) and isinstance(right_cell, str) and left_cell == right_cell:
                return True

        return False


class MCTSNode:
    def __init__(self, mcts, state):
        self.mcts = mcts
        self.state = state
        self.key = mcts.state_key(state)

        self.children = []
        self.parents = set()

        self.total_reward = 0.0
        self.visits = 0

        # For deterministic coverage of action space per node.
        self.actions_left = self.generate_base_actions()

    def avg_reward(self):
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    def update(self, reward):
        self.visits += 1
        self.total_reward += reward

    def generate_base_actions(self):
        """
        Each action changes one kind assignment:
        - direct placement in one database
        - embedding into another kind in one database

        Action tuple format:
        (mode, kind_index, db_index, target_kind_name_or_none)
        """

        actions = []
        mongodb_index = self.get_mongodb_index()

        for kind_index, kind_name in enumerate(self.mcts.kinds):
            for db_index in range(len(self.mcts.databases)):
                actions.append(("direct", kind_index, db_index, None))

                # Embeddings are only allowed in MongoDB.
                # The target host kind must already exist in MongoDB in the current state.
                if mongodb_index is not None and db_index == mongodb_index:
                    for target_kind_index, target_kind in enumerate(self.mcts.kinds):
                        if target_kind == kind_name:
                            continue

                        target_cell_in_mongo = self.state[mongodb_index][target_kind_index]
                        if target_cell_in_mongo is False:
                            continue
                        

                        actions.append(("embed", kind_index, db_index, target_kind))

        random.shuffle(actions)
        return actions

    def get_mongodb_index(self):
        for index, db_name in enumerate(self.mcts.databases):
            if isinstance(db_name, str) and db_name.lower() == "mongodb":
                return index
        return None

    def apply_action(self, action):
        mode, kind_index, db_index, target_kind = action
        moved_kind_name = self.mcts.kinds[kind_index]

        rows = [list(row) for row in self.state]

        # Reset this kind column across all databases first (single assignment policy).
        for row_index in range(len(rows)):
            rows[row_index][kind_index] = False

        if mode == "direct":
            rows[db_index][kind_index] = True
        elif mode == "embed":
            rows[db_index][kind_index] = target_kind
        else:
            raise ValueError("Unknown action mode")

        # If a host kind moved away from a database, remove stale embeddings that pointed to it
        # in those other databases. We keep those kinds in place as direct storage.
        for row_index in range(len(rows)):
            if row_index == db_index:
                continue

            for other_kind_index in range(len(self.mcts.kinds)):
                if rows[row_index][other_kind_index] == moved_kind_name:
                    rows[row_index][other_kind_index] = True

        return tuple(tuple(row) for row in rows)

    def get_viable_actions(self, path_state_keys):
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

    def is_fully_expanded(self, path_state_keys):
        return len(self.get_viable_actions(path_state_keys)) == 0

    def is_leaf(self, path_state_keys):
        for child in self.children:
            if child.key not in path_state_keys:
                return False
        return True

    def expand(self, path_state_keys):
        viable_actions = self.get_viable_actions(path_state_keys)
        if not viable_actions:
            raise RuntimeError("expand called on node without viable actions")

        action = random.choice(viable_actions)
        self.actions_left.remove(action)

        new_state = self.apply_action(action)
        child, is_new = self.mcts.get_or_create_node(new_state)

        # Link parent/child graph edges.
        if child not in self.children:
            self.children.append(child)
        child.parents.add(self)

        return child, is_new

    def select_best_child(self, path_state_keys):
        viable_children = [child for child in self.children if child.key not in path_state_keys]
        if not viable_children:
            raise RuntimeError("select_best_child called with no viable child")

        best_score = None
        best_candidates = []

        for child in viable_children:
            edge_key = (self.key, child.key)
            edge_visits = self.mcts.edge_visits.get(edge_key, 0)

            # UCT2 = avg(child) + C * sqrt(ln(parent_visits) / edge_visits)
            if edge_visits == 0:
                score = float("inf")
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
