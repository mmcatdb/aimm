from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
import random
from typing import Any

from search.mcts import (
    Assignment,
    AssignmentConditions,
    CanExecute,
    CostBreakdown,
    DatabaseId,
    DatabaseInstance,
    LatencyEstimator,
    LatencyKey,
    PrecomputedLatencyEstimator,
    QueryId,
    StorageCostEstimator,
    StorageCostKey,
    TableId,
    WorkloadQuery,
)

PlacementState = tuple[bool, ...]
PlacementPair = tuple[TableId, DatabaseId]
PlacementEdgeKey = tuple[PlacementState, PlacementState]
TablePlacement = dict[TableId, set[DatabaseId]]
DatabasePlacement = dict[DatabaseId, set[TableId]]
TablePlacementInput = Mapping[TableId, Iterable[DatabaseId]] | Iterable[PlacementPair]
QueryDatabaseStorageResolver = Callable[[WorkloadQuery, DatabaseInstance], Iterable[TableId]]
CanStore = Callable[[TableId, DatabaseInstance], bool]
PlacementSchemaFormatter = Callable[[DatabasePlacement], Any]
LatencyEstimatorInput = (
    LatencyEstimator
    | Callable[[WorkloadQuery, DatabaseInstance], float]
)

ACTION_SELECTION_UCT = 'uct'
ACTION_SELECTION_RANDOM = 'random'
ACTION_SELECTION_CHOICES = (
    ACTION_SELECTION_UCT,
    ACTION_SELECTION_RANDOM,
)


@dataclass(frozen=True)
class TablePlacementAction:
    kind: str
    query_id: QueryId | None = None
    from_database_id: DatabaseId | None = None
    to_database_id: DatabaseId | None = None
    table_id: TableId | None = None
    database_id: DatabaseId | None = None


@dataclass(frozen=True)
class TablePlacementOptimizationResult:
    best_assignment: Assignment
    best_placement: DatabasePlacement
    best_cost: float
    best_reward: float
    initial_assignment: Assignment
    initial_placement: DatabasePlacement
    initial_cost: float
    iterations_completed: int
    number_of_unique_states: int
    best_cost_over_time: list[float] = field(default_factory=list)
    best_latency_cost: float = 0.0
    best_storage_cost: float = 0.0
    initial_latency_cost: float = 0.0
    initial_storage_cost: float = 0.0


@dataclass
class TablePlacementNode:
    state: PlacementState
    visits: int = 0
    total_reward: float = 0.0
    expanded_actions: set[TablePlacementAction] = field(default_factory=set)
    children: dict[TablePlacementAction, 'TablePlacementNode'] = field(default_factory=dict)
    candidate_action_states: list[tuple[TablePlacementAction, PlacementState]] | None = None
    cost: float | None = None
    reward: float | None = None

    @property
    def average_reward(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits


class TablePlacementMCTSOptimizer:
    """MCTS over physical table/storage-item placements with derived query routing."""

    def __init__(
        self,
        queries: Sequence[WorkloadQuery],
        databases: Sequence[DatabaseInstance],
        latency_estimator: LatencyEstimatorInput,
        can_execute: CanExecute | None = None,
        exploration_constant: float = math.sqrt(2.0),
        epsilon: float = 1e-12,
        random_seed: int | None = None,
        cache_latencies: bool = True,
        estimate_storage_cost: StorageCostEstimator | None = None,
        get_query_database_storage_table_ids: QueryDatabaseStorageResolver | None = None,
        can_store: CanStore | None = None,
        storage_table_ids: Iterable[TableId] | None = None,
        latency_cost_weight: float = 1.0,
        storage_cost_weight: float = 0.0,
        cache_storage_costs: bool = True,
        assignment_conditions: AssignmentConditions | None = None,
        verbose: bool = False,
        verbose_progress_interval: int = 1000,
        format_placement_schema: PlacementSchemaFormatter | None = None,
        action_selection: str = ACTION_SELECTION_UCT,
    ):
        self.queries = tuple(queries)
        self.databases = tuple(databases)
        self.latency_estimator = latency_estimator
        self.can_execute_fn = can_execute
        self.estimate_storage_cost_fn = estimate_storage_cost
        self.get_query_database_storage_table_ids_fn = get_query_database_storage_table_ids
        self.can_store_fn = can_store
        self.extra_storage_table_ids = (
            self._normalize_storage_table_ids(storage_table_ids, 'storage_table_ids')
            if storage_table_ids is not None
            else frozenset()
        )
        self.exploration_constant = self._validate_non_negative_finite(
            exploration_constant,
            'exploration_constant',
        )
        self.epsilon = self._validate_positive_finite(epsilon, 'epsilon')
        self.latency_cost_weight = self._validate_non_negative_finite(
            latency_cost_weight,
            'latency_cost_weight',
        )
        self.storage_cost_weight = self._validate_non_negative_finite(
            storage_cost_weight,
            'storage_cost_weight',
        )
        self.random = random.Random(random_seed)
        self.cache_latencies = cache_latencies
        self.cache_storage_costs = cache_storage_costs
        self.assignment_conditions = assignment_conditions or AssignmentConditions()
        if action_selection not in ACTION_SELECTION_CHOICES:
            choices = ', '.join(repr(choice) for choice in ACTION_SELECTION_CHOICES)
            raise ValueError(f'action_selection must be one of: {choices}')
        self.action_selection = action_selection
        self.verbose = verbose
        self.verbose_progress_interval = verbose_progress_interval
        self.format_placement_schema_fn = format_placement_schema

        self.query_ids = tuple(query.id for query in self.queries)
        self.database_ids = tuple(database.id for database in self.databases)
        self.query_by_id = {query.id: query for query in self.queries}
        self.database_by_id = {database.id: database for database in self.databases}

        self.required_storage_ids_by_query_database: dict[
            tuple[QueryId, DatabaseId],
            frozenset[TableId],
        ] = {}
        self.placement_keys: tuple[PlacementPair, ...] = ()
        self.placement_index_by_pair: dict[PlacementPair, int] = {}
        self.required_pair_indexes_by_query_database: dict[
            tuple[QueryId, DatabaseId],
            tuple[int, ...] | None,
        ] = {}
        self.query_ids_by_required_pair: dict[PlacementPair, frozenset[QueryId]] = {}
        self.query_ids_by_required_pair_index: dict[int, frozenset[QueryId]] = {}

        self.feasibility_cache: dict[tuple[QueryId, DatabaseId], bool] = {}
        self.latency_cache: dict[LatencyKey, float] = {}
        self.storage_cost_cache: dict[StorageCostKey, float] = {}
        self.state_cost_cache: dict[PlacementState, float] = {}
        self.state_cost_breakdown_cache: dict[PlacementState, CostBreakdown] = {}
        self.state_reward_cache: dict[PlacementState, float] = {}
        self.state_assignment_cache: dict[PlacementState, Assignment | None] = {}
        self.nodes_by_state: dict[PlacementState, TablePlacementNode] = {}
        self.edge_visits: dict[PlacementEdgeKey, int] = {}

        self._baseline_cost = 0.0
        self._best_state: PlacementState | None = None
        self._best_cost = math.inf
        self._best_reward = 0.0
        self._best_latency_cost = 0.0
        self._best_storage_cost = 0.0

        self._validate_inputs()
        self._validate_assignment_conditions()
        self._initialize_required_storage_ids()
        self._initialize_placement_keys()
        self._initialize_required_pair_indexes()
        self._initialize_queries_by_required_pair()
        self._validate_feasible_database_placements_exist()
        if isinstance(latency_estimator, PrecomputedLatencyEstimator):
            latency_estimator.validate_complete(
                self.queries,
                self.databases,
                self._can_execute,
            )

    def optimize(
        self,
        iterations: int,
        initial_placement: TablePlacementInput | None = None,
        collect_trace: bool = True,
    ) -> TablePlacementOptimizationResult:
        if iterations < 0:
            raise ValueError('iterations must be non-negative')

        self._reset_search_state()

        root_state = self._initial_state(initial_placement)
        root = self._get_or_create_node(root_state)

        initial_breakdown = self._compute_state_cost_breakdown(root_state)
        initial_cost = initial_breakdown.total_cost
        self._baseline_cost = initial_cost
        initial_reward = self._reward_for_cost(initial_cost)
        self._cache_state_evaluation(root, initial_breakdown, initial_reward)
        self._consider_best_state(
            root_state,
            initial_breakdown,
            initial_reward,
            report_verbose=False,
        )
        self._print_verbose_best_state(root_state, initial_cost)

        best_cost_over_time = [self._best_cost] if collect_trace else []
        iterations_completed = 0

        for _ in range(iterations):
            path_nodes = [root]
            path_edges: list[PlacementEdgeKey] = []
            path_states = {root.state}
            node = root
            unexpanded_candidates = self._unexpanded_action_candidates(node, path_states)

            while not unexpanded_candidates:
                child = self._select_best_child(node, path_states)
                if child is None:
                    break

                edge = (node.state, child.state)
                path_edges.append(edge)
                path_nodes.append(child)
                path_states.add(child.state)
                node = child
                unexpanded_candidates = self._unexpanded_action_candidates(node, path_states)

            if not unexpanded_candidates:
                reward = self._evaluate_node(node)
            else:
                action, child_state = self.random.choice(unexpanded_candidates)
                child = self._expand(node, action, child_state)
                edge = (node.state, child.state)
                path_edges.append(edge)
                path_nodes.append(child)
                path_states.add(child.state)
                reward = self._evaluate_node(child)

            for visited_node in path_nodes:
                visited_node.visits += 1
                visited_node.total_reward += reward

            for edge in path_edges:
                self.edge_visits[edge] = self.edge_visits.get(edge, 0) + 1

            iterations_completed += 1
            if collect_trace:
                best_cost_over_time.append(self._best_cost)
            self._print_verbose_progress(iterations_completed)

        best_state = self._best_state or root_state
        return TablePlacementOptimizationResult(
            best_assignment=self._state_to_assignment(best_state),
            best_placement=self._state_to_placement_by_database(best_state),
            best_cost=self._best_cost,
            best_reward=self._best_reward,
            initial_assignment=self._state_to_assignment(root_state),
            initial_placement=self._state_to_placement_by_database(root_state),
            initial_cost=initial_cost,
            iterations_completed=iterations_completed,
            number_of_unique_states=len(self.nodes_by_state),
            best_cost_over_time=best_cost_over_time,
            best_latency_cost=self._best_latency_cost,
            best_storage_cost=self._best_storage_cost,
            initial_latency_cost=initial_breakdown.latency_cost,
            initial_storage_cost=initial_breakdown.storage_cost,
        )

    def _reset_search_state(self):
        self.state_cost_cache.clear()
        self.state_cost_breakdown_cache.clear()
        self.state_reward_cache.clear()
        self.state_assignment_cache.clear()
        self.nodes_by_state.clear()
        self.edge_visits.clear()
        self._baseline_cost = 0.0
        self._best_state = None
        self._best_cost = math.inf
        self._best_reward = 0.0
        self._best_latency_cost = 0.0
        self._best_storage_cost = 0.0

    def _validate_inputs(self):
        self._validate_unique_ids(self.query_ids, 'query')
        self._validate_unique_ids(self.database_ids, 'database')

        for query in self.queries:
            self._validate_non_negative_finite(query.weight, f'weight for query {query.id!r}')

        if self.queries and not self.databases:
            raise ValueError('At least one database is required when queries are provided')

        if self.latency_cost_weight == 0 and self.storage_cost_weight == 0:
            raise ValueError(
                'At least one of latency_cost_weight or storage_cost_weight must be positive'
            )

        if self.storage_cost_weight > 0 and self.estimate_storage_cost_fn is None:
            raise ValueError('estimate_storage_cost is required when storage_cost_weight is positive')

        if not isinstance(self.verbose_progress_interval, int):
            raise ValueError('verbose_progress_interval must be an integer')
        if self.verbose_progress_interval < 0:
            raise ValueError('verbose_progress_interval must be non-negative')

    def _validate_assignment_conditions(self):
        must_assign = self.assignment_conditions.must_assign or {}
        must_not_assign = self.assignment_conditions.must_not_assign or {}

        for query_id, database_id in must_assign.items():
            self._validate_assignment_condition_id(query_id, 'query')
            self._validate_assignment_condition_id(database_id, 'database')
            if query_id not in self.query_by_id:
                raise ValueError(
                    f'Assignment condition references unknown query id {query_id!r}'
                )
            if database_id not in self.database_by_id:
                raise ValueError(
                    f'Assignment condition for query {query_id!r} references '
                    f'unknown database id {database_id!r}'
                )

        for query_id, database_ids in must_not_assign.items():
            self._validate_assignment_condition_id(query_id, 'query')
            if query_id not in self.query_by_id:
                raise ValueError(
                    f'Assignment condition references unknown query id {query_id!r}'
                )
            for database_id in database_ids:
                self._validate_assignment_condition_id(database_id, 'database')
                if database_id not in self.database_by_id:
                    raise ValueError(
                        f'Assignment condition for query {query_id!r} references '
                        f'unknown database id {database_id!r}'
                    )

        for query_id, required_database_id in must_assign.items():
            forbidden_database_ids = must_not_assign.get(query_id, frozenset())
            if required_database_id in forbidden_database_ids:
                raise ValueError(
                    f'Assignment conditions require query {query_id!r} to use '
                    f'database {required_database_id!r}, but also forbid that assignment'
                )

    @staticmethod
    def _validate_assignment_condition_id(value: str, label: str):
        if not value:
            raise ValueError(f'Assignment condition {label} id must not be empty')

    def _initialize_required_storage_ids(self):
        self.required_storage_ids_by_query_database = {}
        for query in self.queries:
            for database in self.databases:
                if self.get_query_database_storage_table_ids_fn is not None:
                    table_ids = self.get_query_database_storage_table_ids_fn(query, database)
                else:
                    table_ids = query.storage_table_ids
                self.required_storage_ids_by_query_database[(query.id, database.id)] = (
                    self._normalize_storage_table_ids(
                        table_ids,
                        f'storage table ids for query {query.id!r} on database {database.id!r}',
                    )
                )

    def _initialize_placement_keys(self):
        table_ids = set(self.extra_storage_table_ids)
        for required_ids in self.required_storage_ids_by_query_database.values():
            table_ids.update(required_ids)

        placement_keys = []
        for table_id in sorted(table_ids, key=repr):
            for database in self.databases:
                if self._can_store(table_id, database):
                    placement_keys.append((table_id, database.id))

        self.placement_keys = tuple(placement_keys)
        self.placement_index_by_pair = {
            pair: index
            for index, pair in enumerate(self.placement_keys)
        }

    def _initialize_required_pair_indexes(self):
        self.required_pair_indexes_by_query_database = {}
        for (query_id, database_id), required_ids in (
            self.required_storage_ids_by_query_database.items()
        ):
            indexes: list[int] = []
            for table_id in required_ids:
                index = self.placement_index_by_pair.get((table_id, database_id))
                if index is None:
                    self.required_pair_indexes_by_query_database[(query_id, database_id)] = None
                    break
                indexes.append(index)
            else:
                self.required_pair_indexes_by_query_database[(query_id, database_id)] = (
                    tuple(sorted(indexes))
                )

    def _initialize_queries_by_required_pair(self):
        queries_by_index: dict[int, set[QueryId]] = {}
        for query in self.queries:
            for database in self.databases:
                if not self._can_execute(query, database):
                    continue
                required_indexes = self._query_required_pair_indexes(query, database)
                if required_indexes is None:
                    continue
                for index in required_indexes:
                    queries_by_index.setdefault(index, set()).add(query.id)

        self.query_ids_by_required_pair_index = {
            index: frozenset(query_ids)
            for index, query_ids in queries_by_index.items()
        }

        self.query_ids_by_required_pair = {
            self.placement_keys[index]: query_ids
            for index, query_ids in self.query_ids_by_required_pair_index.items()
        }

    def _validate_feasible_database_placements_exist(self):
        for query in self.queries:
            if not any(
                self._can_execute(query, database)
                and self._query_required_pair_indexes(query, database) is not None
                for database in self.databases
            ):
                raise ValueError(
                    f'No feasible storage placement found for query {query.id!r}'
                )

    @staticmethod
    def _validate_unique_ids(ids: Sequence[str], label: str):
        seen = set()
        for id_value in ids:
            try:
                hash(id_value)
            except TypeError as exc:
                raise ValueError(f'{label.capitalize()} id must be hashable: {id_value!r}') from exc

            if id_value in seen:
                raise ValueError(f'Duplicate {label} id: {id_value!r}')
            seen.add(id_value)

    @staticmethod
    def _validate_non_negative_finite(value: float, label: str) -> float:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{label} must be a finite non-negative number') from exc

        if not math.isfinite(numeric_value) or numeric_value < 0:
            raise ValueError(f'{label} must be a finite non-negative number')
        return numeric_value

    @staticmethod
    def _validate_positive_finite(value: float, label: str) -> float:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'{label} must be a finite positive number') from exc

        if not math.isfinite(numeric_value) or numeric_value <= 0:
            raise ValueError(f'{label} must be a finite positive number')
        return numeric_value

    @staticmethod
    def _normalize_storage_table_ids(
        table_ids: Iterable[TableId] | None,
        label: str,
    ) -> frozenset[TableId]:
        if table_ids is None:
            return frozenset()

        try:
            normalized = frozenset(table_ids)
        except TypeError as exc:
            raise ValueError(f'{label} must be an iterable of hashable ids') from exc

        for table_id in normalized:
            try:
                hash(table_id)
            except TypeError as exc:
                raise ValueError(
                    f'{label} contains an unhashable storage table id: {table_id!r}'
                ) from exc
        return normalized

    def _initial_state(
        self,
        initial_placement: TablePlacementInput | None,
    ) -> PlacementState:
        if initial_placement is None:
            state = tuple(True for _ in self.placement_keys)
        else:
            state = self._placement_input_to_state(initial_placement)

        self._validate_complete_feasible_state(state, 'Initial placement')
        return state

    def _placement_input_to_state(self, initial_placement: TablePlacementInput) -> PlacementState:
        next_state = [False] * len(self.placement_keys)
        known_table_ids = {pair[0] for pair in self.placement_keys}

        for table_id, database_id in self._iter_initial_placement_pairs(initial_placement):
            if table_id not in known_table_ids:
                raise ValueError(f'Initial placement references unknown storage table id {table_id!r}')
            if database_id not in self.database_by_id:
                raise ValueError(f'Initial placement references unknown database id {database_id!r}')

            pair = (table_id, database_id)
            index = self.placement_index_by_pair.get(pair)
            if index is None:
                raise ValueError(
                    f'Initial placement cannot store table {table_id!r} on database {database_id!r}'
                )
            next_state[index] = True

        return tuple(next_state)

    @staticmethod
    def _iter_initial_placement_pairs(
        initial_placement: TablePlacementInput,
    ) -> Iterable[PlacementPair]:
        if isinstance(initial_placement, Mapping):
            for table_id, database_ids in initial_placement.items():
                if isinstance(database_ids, (str, bytes)):
                    yield table_id, database_ids
                    continue
                for database_id in database_ids:
                    yield table_id, database_id
            return

        for table_id, database_id in initial_placement:
            yield table_id, database_id

    def _validate_complete_feasible_state(self, state: PlacementState, label: str):
        if len(state) != len(self.placement_keys):
            raise ValueError(f'{label} length must match number of table/database placements')

        if not self._is_valid_state(state):
            unroutable = [
                query.id
                for query in self.queries
                if not any(self._query_can_run_on_database(state, query, database)
                           for database in self.databases)
            ]
            if unroutable:
                raise ValueError(
                    f'{label} does not provide a runnable database for query ids: {unroutable}'
                )
            raise ValueError(f'{label} is not a valid table placement state')

    def _can_execute(self, query: WorkloadQuery, database: DatabaseInstance) -> bool:
        key = (query.id, database.id)
        cached = self.feasibility_cache.get(key)
        if cached is not None:
            return cached

        required_database_id = self.assignment_conditions.must_assign.get(query.id)
        forbidden_database_ids = self.assignment_conditions.must_not_assign.get(query.id, frozenset())

        if required_database_id is not None and database.id != required_database_id:
            can_execute = False
        elif database.id in forbidden_database_ids:
            can_execute = False
        elif self.can_execute_fn is not None:
            can_execute = bool(self.can_execute_fn(query, database))
        elif query.feasible_database_ids is not None:
            can_execute = database.id in query.feasible_database_ids
        else:
            can_execute = True

        self.feasibility_cache[key] = can_execute
        return can_execute

    def _can_store(self, table_id: TableId, database: DatabaseInstance) -> bool:
        if self.can_store_fn is None:
            return True
        return bool(self.can_store_fn(table_id, database))

    def _estimate_latency(self, query: WorkloadQuery, database: DatabaseInstance) -> float:
        key = (query.id, database.id)
        if self.cache_latencies and key in self.latency_cache:
            return self.latency_cache[key]

        try:
            if hasattr(self.latency_estimator, 'estimate_latency'):
                latency = float(self.latency_estimator.estimate_latency(query, database))
            else:
                latency = float(self.latency_estimator(query, database))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'Latency estimate for query {query.id!r} on database {database.id!r} '
                'must be a finite non-negative number'
            ) from exc

        if not math.isfinite(latency) or latency < 0:
            raise ValueError(
                f'Latency estimate for query {query.id!r} on database {database.id!r} '
                'must be a finite non-negative number'
            )

        if self.cache_latencies:
            self.latency_cache[key] = latency
        return latency

    def _estimate_storage_cost(self, table_id: TableId, database: DatabaseInstance) -> float:
        key = (table_id, database.id)
        if self.cache_storage_costs and key in self.storage_cost_cache:
            return self.storage_cost_cache[key]

        if self.estimate_storage_cost_fn is None:
            return 0.0

        try:
            storage_cost = float(self.estimate_storage_cost_fn(table_id, database))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'Storage cost estimate for table {table_id!r} on database {database.id!r} '
                'must be a finite non-negative number'
            ) from exc

        if not math.isfinite(storage_cost) or storage_cost < 0:
            raise ValueError(
                f'Storage cost estimate for table {table_id!r} on database {database.id!r} '
                'must be a finite non-negative number'
            )

        if self.cache_storage_costs:
            self.storage_cost_cache[key] = storage_cost
        return storage_cost

    def _get_or_create_node(self, state: PlacementState) -> TablePlacementNode:
        node = self.nodes_by_state.get(state)
        if node is None:
            node = TablePlacementNode(state=state)
            self.nodes_by_state[state] = node
        return node

    def _is_fully_expanded(self, node: TablePlacementNode, path_states: set[PlacementState]) -> bool:
        return not self._unexpanded_action_candidates(node, path_states)

    def _choose_unexpanded_action(
        self,
        node: TablePlacementNode,
        path_states: set[PlacementState],
    ) -> TablePlacementAction | None:
        candidates = self._unexpanded_action_candidates(node, path_states)
        if not candidates:
            return None
        action, _ = self.random.choice(candidates)
        return action

    def _unexpanded_actions(
        self,
        node: TablePlacementNode,
        path_states: set[PlacementState],
    ) -> list[TablePlacementAction]:
        return [
            action
            for action, _ in self._unexpanded_action_candidates(node, path_states)
        ]

    def _unexpanded_action_candidates(
        self,
        node: TablePlacementNode,
        path_states: set[PlacementState],
    ) -> list[tuple[TablePlacementAction, PlacementState]]:
        return [
            (action, next_state)
            for action, next_state in self._candidate_action_states(node)
            if action not in node.expanded_actions
            and next_state not in path_states
        ]

    def _candidate_action_states(
        self,
        node: TablePlacementNode,
    ) -> list[tuple[TablePlacementAction, PlacementState]]:
        if node.candidate_action_states is not None:
            return node.candidate_action_states

        candidates: list[tuple[TablePlacementAction, PlacementState]] = []
        seen_next_states: set[PlacementState] = set()

        for action in self._generate_admissible_actions(node.state):
            next_state = self._apply_action(node.state, action)
            if next_state == node.state:
                continue
            if next_state in seen_next_states:
                continue

            seen_next_states.add(next_state)
            candidates.append((action, next_state))

        node.candidate_action_states = candidates
        return candidates

    def _generate_admissible_actions(
        self,
        state: PlacementState,
    ) -> Iterable[TablePlacementAction]:
        yield from self._generate_add_query_footprint_actions(state)
        yield from self._generate_move_query_footprint_actions(state)
        yield from self._generate_remove_table_placement_actions(state)

    def _generate_add_query_footprint_actions(
        self,
        state: PlacementState,
    ) -> Iterable[TablePlacementAction]:
        for query in self.queries:
            for database in self.databases:
                if not self._can_execute(query, database):
                    continue
                required_indexes = self._query_required_pair_indexes(query, database)
                if required_indexes is None:
                    continue
                if not any(not state[index] for index in required_indexes):
                    continue

                action = TablePlacementAction(
                    kind='add-query-footprint',
                    query_id=query.id,
                    to_database_id=database.id,
                )
                yield action

    def _generate_move_query_footprint_actions(
        self,
        state: PlacementState,
    ) -> Iterable[TablePlacementAction]:
        for query in self.queries:
            for from_database in self.databases:
                if not self._query_can_run_on_database(state, query, from_database):
                    continue
                for to_database in self.databases:
                    if to_database.id == from_database.id:
                        continue
                    if not self._can_execute(query, to_database):
                        continue
                    if self._query_required_pair_indexes(query, to_database) is None:
                        continue

                    action = TablePlacementAction(
                        kind='move-query-footprint',
                        query_id=query.id,
                        from_database_id=from_database.id,
                        to_database_id=to_database.id,
                    )
                    yield action

    def _generate_remove_table_placement_actions(
        self,
        state: PlacementState,
    ) -> Iterable[TablePlacementAction]:
        for index, is_present in enumerate(state):
            if not is_present:
                continue

            table_id, database_id = self.placement_keys[index]
            action = TablePlacementAction(
                kind='remove-table-placement',
                table_id=table_id,
                database_id=database_id,
            )
            yield action

    def _apply_action(
        self,
        state: PlacementState,
        action: TablePlacementAction,
    ) -> PlacementState:
        if action.kind == 'add-query-footprint':
            return self._apply_add_query_footprint(state, action)
        if action.kind == 'move-query-footprint':
            return self._apply_move_query_footprint(state, action)
        if action.kind == 'remove-table-placement':
            return self._apply_remove_table_placement(state, action)
        raise ValueError(f'Unknown table placement action kind {action.kind!r}')

    def _apply_add_query_footprint(
        self,
        state: PlacementState,
        action: TablePlacementAction,
    ) -> PlacementState:
        query = self._action_query(action)
        database = self._action_to_database(action)
        required_indexes = self._require_query_required_pair_indexes(query, database)

        next_state = list(state)
        for index in required_indexes:
            next_state[index] = True
        return tuple(next_state)

    def _apply_move_query_footprint(
        self,
        state: PlacementState,
        action: TablePlacementAction,
    ) -> PlacementState:
        query = self._action_query(action)
        from_database = self._action_from_database(action)
        to_database = self._action_to_database(action)
        source_indexes = self._require_query_required_pair_indexes(query, from_database)
        target_indexes = self._require_query_required_pair_indexes(query, to_database)

        next_state = list(state)
        for index in target_indexes:
            next_state[index] = True

        candidate_state = tuple(next_state)
        for index in source_indexes:
            if not candidate_state[index]:
                continue

            removal_state = self._remove_index_if_safe(candidate_state, index)
            if removal_state != candidate_state:
                candidate_state = removal_state

        return candidate_state

    def _apply_remove_table_placement(
        self,
        state: PlacementState,
        action: TablePlacementAction,
    ) -> PlacementState:
        if action.table_id is None or action.database_id is None:
            raise ValueError('remove-table-placement action requires table_id and database_id')

        pair = (action.table_id, action.database_id)
        index = self.placement_index_by_pair.get(pair)
        if index is None:
            raise ValueError(
                f'Cannot remove unknown table/database placement {pair!r}'
            )
        if not state[index]:
            return state

        return self._remove_index_if_safe(state, index)

    def _remove_pair_if_safe(
        self,
        state: PlacementState,
        pair: PlacementPair,
    ) -> PlacementState:
        index = self.placement_index_by_pair[pair]
        return self._remove_index_if_safe(state, index)

    def _remove_index_if_safe(
        self,
        state: PlacementState,
        index: int,
    ) -> PlacementState:
        if not state[index]:
            return state

        next_state = list(state)
        next_state[index] = False
        candidate_state = tuple(next_state)
        if self._affected_queries_routable_after_removal(candidate_state, index):
            return candidate_state
        return state

    def _affected_queries_routable_after_removal(
        self,
        candidate_state: PlacementState,
        removed_index: int,
    ) -> bool:
        affected_query_ids = self.query_ids_by_required_pair_index.get(removed_index, frozenset())
        for query_id in affected_query_ids:
            query = self.query_by_id[query_id]
            if not any(
                self._query_can_run_on_database(candidate_state, query, database)
                for database in self.databases
            ):
                return False
        return True

    def _action_query(self, action: TablePlacementAction) -> WorkloadQuery:
        if action.query_id is None:
            raise ValueError(f'{action.kind} action requires query_id')
        query = self.query_by_id.get(action.query_id)
        if query is None:
            raise ValueError(f'Action references unknown query id {action.query_id!r}')
        return query

    def _action_from_database(self, action: TablePlacementAction) -> DatabaseInstance:
        if action.from_database_id is None:
            raise ValueError(f'{action.kind} action requires from_database_id')
        database = self.database_by_id.get(action.from_database_id)
        if database is None:
            raise ValueError(
                f'Action references unknown source database id {action.from_database_id!r}'
            )
        return database

    def _action_to_database(self, action: TablePlacementAction) -> DatabaseInstance:
        if action.to_database_id is None:
            raise ValueError(f'{action.kind} action requires to_database_id')
        database = self.database_by_id.get(action.to_database_id)
        if database is None:
            raise ValueError(
                f'Action references unknown target database id {action.to_database_id!r}'
            )
        return database

    def _expand(
        self,
        node: TablePlacementNode,
        action: TablePlacementAction,
        child_state: PlacementState | None = None,
    ) -> TablePlacementNode:
        next_state = child_state if child_state is not None else self._apply_action(node.state, action)
        child = self._get_or_create_node(next_state)
        node.expanded_actions.add(action)
        node.children[action] = child
        return child

    def _select_best_child(
        self,
        node: TablePlacementNode,
        path_states: set[PlacementState],
    ) -> TablePlacementNode | None:
        candidates = [child for child in node.children.values() if child.state not in path_states]
        if not candidates:
            return None

        if self.action_selection == ACTION_SELECTION_RANDOM:
            return self.random.choice(candidates)

        best_score = -math.inf
        best_children: list[TablePlacementNode] = []
        for child in candidates:
            edge_visits = self.edge_visits.get((node.state, child.state), 0)
            if child.visits == 0 or edge_visits == 0:
                score = math.inf
            else:
                exploration = math.sqrt(
                    math.log(node.visits + 1) / (edge_visits + self.epsilon)
                )
                score = child.average_reward + self.exploration_constant * exploration

            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)

        return self.random.choice(best_children)

    def _evaluate_node(self, node: TablePlacementNode) -> float:
        cached_reward = self.state_reward_cache.get(node.state)
        if cached_reward is not None:
            node.cost = self.state_cost_cache[node.state]
            node.reward = cached_reward
            return cached_reward

        breakdown = self._compute_state_cost_breakdown(node.state)
        reward = self._reward_for_cost(breakdown.total_cost)
        self._cache_state_evaluation(node, breakdown, reward)
        self._consider_best_state(node.state, breakdown, reward)
        return reward

    def _compute_state_cost(self, state: PlacementState) -> float:
        return self._compute_state_cost_breakdown(state).total_cost

    def _compute_state_cost_breakdown(self, state: PlacementState) -> CostBreakdown:
        cached_breakdown = self.state_cost_breakdown_cache.get(state)
        if cached_breakdown is not None:
            return cached_breakdown

        assignment = self._derive_assignment(state)
        if assignment is None:
            breakdown = CostBreakdown(
                total_cost=math.inf,
                latency_cost=math.inf,
                storage_cost=math.inf,
            )
            self.state_cost_cache[state] = math.inf
            self.state_cost_breakdown_cache[state] = breakdown
            self.state_reward_cache[state] = 0.0
            return breakdown

        latency_cost = self._compute_assignment_latency_cost(assignment)
        storage_cost = self._compute_state_storage_cost(state)
        total_cost = (
            self.latency_cost_weight * latency_cost
            + self.storage_cost_weight * storage_cost
        )

        breakdown = CostBreakdown(
            total_cost=total_cost,
            latency_cost=latency_cost,
            storage_cost=storage_cost,
        )
        self.state_cost_cache[state] = total_cost
        self.state_cost_breakdown_cache[state] = breakdown
        return breakdown

    def _derive_assignment(self, state: PlacementState) -> Assignment | None:
        if state in self.state_assignment_cache:
            return self.state_assignment_cache[state]

        if len(state) != len(self.placement_keys):
            self.state_assignment_cache[state] = None
            return None

        assignment: Assignment = {}
        for query in self.queries:
            candidate_databases = [
                database
                for database in self.databases
                if self._query_can_run_on_database(state, query, database)
            ]
            if not candidate_databases:
                self.state_assignment_cache[state] = None
                return None

            best_database = min(
                candidate_databases,
                key=lambda database: self._estimate_latency(query, database),
            )
            assignment[query.id] = best_database.id

        self.state_assignment_cache[state] = assignment
        return assignment

    def _compute_assignment_latency_cost(self, assignment: Assignment) -> float:
        total_cost = 0.0
        for query in self.queries:
            database = self.database_by_id[assignment[query.id]]
            total_cost += float(query.weight) * self._estimate_latency(query, database)
        return total_cost

    def _compute_state_storage_cost(self, state: PlacementState) -> float:
        if self.estimate_storage_cost_fn is None:
            return 0.0

        total_cost = 0.0
        for is_present, (table_id, database_id) in zip(state, self.placement_keys):
            if not is_present:
                continue
            database = self.database_by_id[database_id]
            total_cost += self._estimate_storage_cost(table_id, database)
        return total_cost

    def _is_valid_state(self, state: PlacementState) -> bool:
        return len(state) == len(self.placement_keys) and self._all_queries_routable(state)

    def _all_queries_routable(self, state: PlacementState) -> bool:
        return all(
            any(self._query_can_run_on_database(state, query, database)
                for database in self.databases)
            for query in self.queries
        )

    def _query_can_run_on_database(
        self,
        state: PlacementState,
        query: WorkloadQuery,
        database: DatabaseInstance,
    ) -> bool:
        if not self._can_execute(query, database):
            return False

        required_indexes = self._query_required_pair_indexes(query, database)
        if required_indexes is None:
            return False

        for index in required_indexes:
            if not state[index]:
                return False
        return True

    def _query_required_pair_indexes(
        self,
        query: WorkloadQuery,
        database: DatabaseInstance,
    ) -> tuple[int, ...] | None:
        return self.required_pair_indexes_by_query_database[(query.id, database.id)]

    def _query_required_pairs(
        self,
        query: WorkloadQuery,
        database: DatabaseInstance,
    ) -> tuple[PlacementPair, ...] | None:
        indexes = self._query_required_pair_indexes(query, database)
        if indexes is None:
            return None
        return tuple(self.placement_keys[index] for index in indexes)

    def _require_query_required_pair_indexes(
        self,
        query: WorkloadQuery,
        database: DatabaseInstance,
    ) -> tuple[int, ...]:
        indexes = self._query_required_pair_indexes(query, database)
        if indexes is None:
            raise ValueError(
                f'No storage placement can satisfy query {query.id!r} on database {database.id!r}'
            )
        return indexes

    def _require_query_required_pairs(
        self,
        query: WorkloadQuery,
        database: DatabaseInstance,
    ) -> tuple[PlacementPair, ...]:
        pairs = self._query_required_pairs(query, database)
        if pairs is None:
            raise ValueError(
                f'No storage placement can satisfy query {query.id!r} on database {database.id!r}'
            )
        return pairs

    def _reward_for_cost(self, cost: float) -> float:
        if not math.isfinite(cost):
            return 0.0
        if self._baseline_cost == 0:
            return 1.0 if cost == 0 else 0.0
        return self._baseline_cost / max(cost, self.epsilon)

    def _cache_state_evaluation(
        self,
        node: TablePlacementNode,
        breakdown: CostBreakdown,
        reward: float,
    ):
        node.cost = breakdown.total_cost
        node.reward = reward
        self.state_cost_cache[node.state] = breakdown.total_cost
        self.state_cost_breakdown_cache[node.state] = breakdown
        self.state_reward_cache[node.state] = reward

    def _consider_best_state(
        self,
        state: PlacementState,
        breakdown: CostBreakdown,
        reward: float,
        report_verbose: bool = True,
    ):
        if breakdown.total_cost < self._best_cost:
            self._best_state = state
            self._best_cost = breakdown.total_cost
            self._best_reward = reward
            self._best_latency_cost = breakdown.latency_cost
            self._best_storage_cost = breakdown.storage_cost
            if report_verbose:
                self._print_verbose_best_state(state, breakdown.total_cost)

    def _print_verbose_best_state(self, state: PlacementState, cost: float):
        if not self.verbose:
            return

        print(json.dumps({
            'type': 'best-state',
            'iteration': len(self.nodes_by_state),
            'states': len(self.nodes_by_state),
            'state': self._format_verbose_schema(state),
            'cost': cost,
        }), flush=True)

    def _print_verbose_progress(self, iterations_completed: int):
        if (
            not self.verbose
            or self.verbose_progress_interval == 0
            or iterations_completed % self.verbose_progress_interval != 0
        ):
            return

        print(json.dumps({
            'type': 'progress',
            'iteration': iterations_completed,
            'states': len(self.nodes_by_state),
        }), flush=True)

    def _format_verbose_schema(self, state: PlacementState) -> Any:
        placement = self._state_to_placement_by_database(state)
        if self.format_placement_schema_fn is not None:
            return self.format_placement_schema_fn(placement)

        return {
            database_id: sorted(table_ids, key=repr)
            for database_id, table_ids in placement.items()
        }

    def _state_to_assignment(self, state: PlacementState) -> Assignment:
        assignment = self._derive_assignment(state)
        if assignment is None:
            return {}
        return dict(assignment)

    def _state_to_placement_by_database(self, state: PlacementState) -> DatabasePlacement:
        placement: DatabasePlacement = {}
        for is_present, (table_id, database_id) in zip(state, self.placement_keys):
            if is_present:
                placement.setdefault(database_id, set()).add(table_id)
        return placement

    def _state_to_placement_by_table(self, state: PlacementState) -> TablePlacement:
        placement: TablePlacement = {}
        for is_present, (table_id, database_id) in zip(state, self.placement_keys):
            if is_present:
                placement.setdefault(table_id, set()).add(database_id)
        return placement


TablePlacementMCTS = TablePlacementMCTSOptimizer
