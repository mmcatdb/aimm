from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import math
import random
from typing import Any


QueryId = str
DatabaseId = str
TableId = str
State = tuple[DatabaseId, ...]
StateKey = State
StateMapping = list[DatabaseId]
Assignment = dict[QueryId, DatabaseId]
FeasibilityKey = tuple[QueryId, DatabaseId]
LatencyKey = tuple[QueryId, DatabaseId]
StorageCostKey = tuple[TableId, DatabaseId]
EdgeKey = tuple[StateKey, StateKey]

LatencyEstimator = Callable[['WorkloadQuery', 'DatabaseInstance'], float]
CanExecute = Callable[['WorkloadQuery', 'DatabaseInstance'], bool]
StorageCostEstimator = Callable[[TableId, 'DatabaseInstance'], float]
QueryStorageTableResolver = Callable[['WorkloadQuery'], Iterable[TableId]]
LatencyEstimateInput = (
    Mapping[LatencyKey, float]
    | Iterable[tuple[QueryId, DatabaseId, float]]
)


@dataclass(frozen=True)
class WorkloadQuery:
    id: QueryId
    weight: float
    payload: Any = None
    feasible_database_ids: Iterable[DatabaseId] | None = None
    storage_table_ids: Iterable[TableId] | None = None

    def __post_init__(self):
        if self.feasible_database_ids is not None:
            object.__setattr__(
                self,
                'feasible_database_ids',
                frozenset(self.feasible_database_ids),
            )
        if self.storage_table_ids is not None:
            object.__setattr__(
                self,
                'storage_table_ids',
                frozenset(self.storage_table_ids),
            )


@dataclass(frozen=True)
class DatabaseInstance:
    id: DatabaseId
    payload: Any = None


class PrecomputedLatencyEstimator:
    """Latency estimator backed by a validated query/database latency matrix."""

    def __init__(self, latencies: LatencyEstimateInput):
        self.latencies = self._normalize_latencies(latencies)

    def estimate_latency(self, query: WorkloadQuery, database: DatabaseInstance) -> float:
        key = (query.id, database.id)
        if key not in self.latencies:
            raise ValueError(
                f'Missing precomputed latency estimate for query {query.id!r} '
                f'on database {database.id!r}'
            )
        return self.latencies[key]

    def validate_complete(
        self,
        queries: Sequence[WorkloadQuery],
        databases: Sequence[DatabaseInstance],
        can_execute: CanExecute,
    ):
        missing = [
            (query.id, database.id)
            for query in queries
            for database in databases
            if can_execute(query, database) and (query.id, database.id) not in self.latencies
        ]
        if missing:
            formatted = ', '.join(f'{query_id}@{database_id}' for query_id, database_id in missing)
            raise ValueError(
                'Missing precomputed latency estimates for required query/database pairs: '
                f'{formatted}'
            )

    @classmethod
    def _normalize_latencies(
        cls,
        latencies: LatencyEstimateInput,
    ) -> dict[LatencyKey, float]:
        normalized: dict[LatencyKey, float] = {}
        for query_id, database_id, latency in cls._iter_latency_entries(latencies):
            key = cls._validate_latency_key(query_id, database_id)
            if key in normalized:
                raise ValueError(
                    f'Duplicate precomputed latency estimate for query {query_id!r} '
                    f'on database {database_id!r}'
                )
            normalized[key] = cls._validate_latency(latency, query_id, database_id)
        return normalized

    @staticmethod
    def _iter_latency_entries(
        latencies: LatencyEstimateInput,
    ) -> Iterable[tuple[QueryId, DatabaseId, float]]:
        if isinstance(latencies, Mapping):
            for key, latency in latencies.items():
                try:
                    query_id, database_id = key
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        'Precomputed latency mapping keys must be '
                        '(query_id, database_id) pairs'
                    ) from exc
                yield query_id, database_id, latency
            return

        for item in latencies:
            try:
                query_id, database_id, latency = item
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    'Precomputed latency entries must be '
                    '(query_id, database_id, latency) triples'
                ) from exc
            yield query_id, database_id, latency

    @staticmethod
    def _validate_latency_key(query_id: QueryId, database_id: DatabaseId) -> LatencyKey:
        for label, value in (('query id', query_id), ('database id', database_id)):
            try:
                hash(value)
            except TypeError as exc:
                raise ValueError(
                    f'Precomputed latency {label} must be hashable: {value!r}'
                ) from exc
        return query_id, database_id

    @staticmethod
    def _validate_latency(
        latency: float,
        query_id: QueryId,
        database_id: DatabaseId,
    ) -> float:
        try:
            numeric_latency = float(latency)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f'Precomputed latency estimate for query {query_id!r} '
                f'on database {database_id!r} must be a finite non-negative number'
            ) from exc

        if not math.isfinite(numeric_latency) or numeric_latency < 0:
            raise ValueError(
                f'Precomputed latency estimate for query {query_id!r} '
                f'on database {database_id!r} must be a finite non-negative number'
            )
        return numeric_latency


@dataclass(frozen=True)
class ReassignQuery:
    query_id: QueryId
    from_database_id: DatabaseId
    to_database_id: DatabaseId


@dataclass(frozen=True)
class OptimizationResult:
    best_assignment: Assignment
    best_cost: float
    best_reward: float
    initial_assignment: Assignment
    initial_cost: float
    iterations_completed: int
    number_of_unique_states: int
    best_cost_over_time: list[float] = field(default_factory=list)
    best_latency_cost: float = 0.0
    best_storage_cost: float = 0.0
    initial_latency_cost: float = 0.0
    initial_storage_cost: float = 0.0


@dataclass(frozen=True)
class CostBreakdown:
    total_cost: float
    latency_cost: float
    storage_cost: float


@dataclass
class Node:
    state: State
    visits: int = 0
    total_reward: float = 0.0
    expanded_actions: set[ReassignQuery] = field(default_factory=set)
    children: dict[ReassignQuery, 'Node'] = field(default_factory=dict)
    cost: float | None = None
    reward: float | None = None

    @property
    def average_reward(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits


class MCTSOptimizer:
    """Graph-aware MCTS for workload-aware query-to-database routing."""

    def __init__(
        self,
        queries: Sequence[WorkloadQuery],
        databases: Sequence[DatabaseInstance],
        estimate_latency: LatencyEstimator | None = None,
        can_execute: CanExecute | None = None,
        exploration_constant: float = math.sqrt(2.0),
        epsilon: float = 1e-12,
        random_seed: int | None = None,
        cache_latencies: bool = True,
        estimate_storage_cost: StorageCostEstimator | None = None,
        get_query_storage_table_ids: QueryStorageTableResolver | None = None,
        latency_cost_weight: float = 1.0,
        storage_cost_weight: float = 0.0,
        cache_storage_costs: bool = True,
        latency_estimates: LatencyEstimateInput | PrecomputedLatencyEstimator | None = None,
    ):
        self.queries = tuple(queries)
        self.databases = tuple(databases)
        self.can_execute_fn = can_execute
        self.estimate_storage_cost_fn = estimate_storage_cost
        self.get_query_storage_table_ids_fn = get_query_storage_table_ids
        self.precomputed_latency_estimator = self._build_precomputed_latency_estimator(
            estimate_latency,
            latency_estimates,
        )
        self.uses_precomputed_latency_estimator = estimate_latency is None
        self.estimate_latency_fn = (
            estimate_latency
            if estimate_latency is not None
            else self.precomputed_latency_estimator.estimate_latency
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

        self.query_ids = tuple(query.id for query in self.queries)
        self.database_ids = tuple(database.id for database in self.databases)
        self.query_by_id = {query.id: query for query in self.queries}
        self.database_by_id = {database.id: database for database in self.databases}
        self.query_index_by_id = {query.id: index for index, query in enumerate(self.queries)}
        self.query_storage_table_ids_by_id: dict[QueryId, frozenset[TableId]] = {}

        self.feasibility_cache: dict[FeasibilityKey, bool] = {}
        self.latency_cache: dict[LatencyKey, float] = {}
        self.storage_cost_cache: dict[StorageCostKey, float] = {}
        self.state_cost_cache: dict[State, float] = {}
        self.state_cost_breakdown_cache: dict[State, CostBreakdown] = {}
        self.state_reward_cache: dict[State, float] = {}
        self.nodes_by_state: dict[State, Node] = {}
        self.edge_visits: dict[EdgeKey, int] = {}

        self._baseline_cost = 0.0
        self._best_state: State | None = None
        self._best_cost = math.inf
        self._best_reward = 0.0
        self._best_latency_cost = 0.0
        self._best_storage_cost = 0.0

        self._validate_inputs()
        self._validate_feasible_databases_exist()
        self._validate_precomputed_latencies_complete()
        self._initialize_query_storage_table_ids()

    def optimize(
        self,
        iterations: int,
        initial_assignment: Mapping[QueryId, DatabaseId] | Sequence[DatabaseId] | None = None,
        collect_trace: bool = True,
    ) -> OptimizationResult:
        if iterations < 0:
            raise ValueError('iterations must be non-negative')

        self._reset_search_state()

        if not self.queries:
            root = self._get_or_create_node(())
            root.cost = 0.0
            root.reward = 1.0
            self.state_cost_cache[root.state] = 0.0
            self.state_reward_cache[root.state] = 1.0
            return OptimizationResult(
                best_assignment={},
                best_cost=0.0,
                best_reward=1.0,
                initial_assignment={},
                initial_cost=0.0,
                iterations_completed=0,
                number_of_unique_states=len(self.nodes_by_state),
                best_cost_over_time=[0.0] if collect_trace else [],
            )

        root_state = self._initial_state(initial_assignment)
        root = self._get_or_create_node(root_state)

        initial_breakdown = self._compute_state_cost_breakdown(root_state)
        initial_cost = initial_breakdown.total_cost
        self._baseline_cost = initial_cost
        initial_reward = self._reward_for_cost(initial_cost)
        self._cache_state_evaluation(root, initial_breakdown, initial_reward)
        self._consider_best_state(root_state, initial_breakdown, initial_reward)

        best_cost_over_time = [self._best_cost] if collect_trace else []
        iterations_completed = 0

        for _ in range(iterations):
            path_nodes = [root]
            path_edges: list[EdgeKey] = []
            path_states = {root.state}
            node = root

            while self._is_fully_expanded(node, path_states):
                child = self._select_best_child(node, path_states)
                if child is None:
                    break

                edge = (node.state, child.state)
                path_edges.append(edge)
                path_nodes.append(child)
                path_states.add(child.state)
                node = child

            if self._is_fully_expanded(node, path_states):
                reward = self._evaluate_node(node)
            else:
                action = self._choose_unexpanded_action(node, path_states)
                if action is None:
                    reward = self._evaluate_node(node)
                else:
                    child = self._expand(node, action)
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

        return OptimizationResult(
            best_assignment=self._state_to_assignment(self._best_state or root_state),
            best_cost=self._best_cost,
            best_reward=self._best_reward,
            initial_assignment=self._state_to_assignment(root_state),
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

    @staticmethod
    def _build_precomputed_latency_estimator(
        estimate_latency: LatencyEstimator | None,
        latency_estimates: LatencyEstimateInput | PrecomputedLatencyEstimator | None,
    ) -> PrecomputedLatencyEstimator:
        if estimate_latency is not None and latency_estimates is not None:
            raise ValueError('Provide either estimate_latency or latency_estimates, not both')

        if latency_estimates is None:
            if estimate_latency is None:
                raise ValueError('estimate_latency or latency_estimates is required')
            return PrecomputedLatencyEstimator({})

        if isinstance(latency_estimates, PrecomputedLatencyEstimator):
            return latency_estimates
        return PrecomputedLatencyEstimator(latency_estimates)

    def _validate_precomputed_latencies_complete(self):
        if self.uses_precomputed_latency_estimator:
            self.precomputed_latency_estimator.validate_complete(
                self.queries,
                self.databases,
                self._can_execute,
            )

    def _initialize_query_storage_table_ids(self):
        if self.estimate_storage_cost_fn is None:
            self.query_storage_table_ids_by_id = {
                query.id: frozenset()
                for query in self.queries
            }
            return

        self.query_storage_table_ids_by_id = {
            query.id: self._resolve_query_storage_table_ids(query)
            for query in self.queries
        }

    def _resolve_query_storage_table_ids(self, query: WorkloadQuery) -> frozenset[TableId]:
        if self.get_query_storage_table_ids_fn is not None:
            table_ids = self.get_query_storage_table_ids_fn(query)
        else:
            table_ids = query.storage_table_ids

        return self._normalize_storage_table_ids(table_ids, query.id)

    @staticmethod
    def _normalize_storage_table_ids(
        table_ids: Iterable[TableId] | None,
        query_id: QueryId,
    ) -> frozenset[TableId]:
        if table_ids is None:
            return frozenset()

        try:
            normalized = frozenset(table_ids)
        except TypeError as exc:
            raise ValueError(
                f'Storage table ids for query {query_id!r} must be an iterable of hashable ids'
            ) from exc

        for table_id in normalized:
            try:
                hash(table_id)
            except TypeError as exc:
                raise ValueError(
                    f'Storage table id for query {query_id!r} must be hashable: {table_id!r}'
                ) from exc

        return normalized

    def _validate_feasible_databases_exist(self):
        for query in self.queries:
            if not any(self._can_execute(query, database) for database in self.databases):
                raise ValueError(f'No feasible database found for query {query.id!r}')

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

    def _initial_state(
        self,
        initial_assignment: Mapping[QueryId, DatabaseId] | Sequence[DatabaseId] | None,
    ) -> State:
        if initial_assignment is None:
            return tuple(self._first_feasible_database(query).id for query in self.queries)

        if isinstance(initial_assignment, Mapping):
            missing = [query.id for query in self.queries if query.id not in initial_assignment]
            extra = [query_id for query_id in initial_assignment if query_id not in self.query_by_id]
            if missing or extra:
                details = []
                if missing:
                    details.append(f'missing query ids: {missing}')
                if extra:
                    details.append(f'unknown query ids: {extra}')
                raise ValueError('Initial assignment must assign every query exactly once (' + '; '.join(details) + ')')

            state = tuple(initial_assignment[query.id] for query in self.queries)
        else:
            if isinstance(initial_assignment, (str, bytes)):
                raise ValueError('Initial assignment sequence must not be a string')
            if len(initial_assignment) != len(self.queries):
                raise ValueError('Initial assignment sequence length must match number of queries')
            state = tuple(initial_assignment)

        self._validate_complete_feasible_state(state, 'Initial assignment')
        return state

    def _first_feasible_database(self, query: WorkloadQuery) -> DatabaseInstance:
        for database in self.databases:
            if self._can_execute(query, database):
                return database
        raise ValueError(f'No feasible database found for query {query.id!r}')

    def _validate_complete_feasible_state(self, state: State, label: str):
        if len(state) != len(self.queries):
            raise ValueError(f'{label} must assign every query exactly once')

        for query, database_id in zip(self.queries, state):
            database = self.database_by_id.get(database_id)
            if database is None:
                raise ValueError(f'{label} references unknown database id {database_id!r}')
            if not self._can_execute(query, database):
                raise ValueError(
                    f'{label} assigns query {query.id!r} to infeasible database {database_id!r}'
                )

    def _can_execute(self, query: WorkloadQuery, database: DatabaseInstance) -> bool:
        key = (query.id, database.id)
        cached = self.feasibility_cache.get(key)
        if cached is not None:
            return cached

        if self.can_execute_fn is not None:
            can_execute = bool(self.can_execute_fn(query, database))
        elif query.feasible_database_ids is not None:
            can_execute = database.id in query.feasible_database_ids
        else:
            can_execute = True

        self.feasibility_cache[key] = can_execute
        return can_execute

    def _estimate_latency(self, query: WorkloadQuery, database: DatabaseInstance) -> float:
        key = (query.id, database.id)
        if self.cache_latencies and key in self.latency_cache:
            return self.latency_cache[key]

        try:
            latency = float(self.estimate_latency_fn(query, database))
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

    def _get_or_create_node(self, state: State) -> Node:
        node = self.nodes_by_state.get(state)
        if node is None:
            node = Node(state=state)
            self.nodes_by_state[state] = node
        return node

    def _is_fully_expanded(self, node: Node, path_states: set[State]) -> bool:
        return not self._unexpanded_actions(node, path_states)

    def _choose_unexpanded_action(self, node: Node, path_states: set[State]) -> ReassignQuery | None:
        actions = self._unexpanded_actions(node, path_states)
        if not actions:
            return None
        return self.random.choice(actions)

    def _unexpanded_actions(self, node: Node, path_states: set[State]) -> list[ReassignQuery]:
        return [
            action
            for action in self._generate_admissible_actions(node.state)
            if action not in node.expanded_actions
            and self._apply_action(node.state, action) not in path_states
        ]

    def _generate_admissible_actions(self, state: State) -> Iterable[ReassignQuery]:
        for query_index, query in enumerate(self.queries):
            current_database_id = state[query_index]
            for database in self.databases:
                if database.id == current_database_id:
                    continue
                if self._can_execute(query, database):
                    yield ReassignQuery(
                        query_id=query.id,
                        from_database_id=current_database_id,
                        to_database_id=database.id,
                    )

    def _apply_action(self, state: State, action: ReassignQuery) -> State:
        query_index = self.query_index_by_id[action.query_id]
        if state[query_index] != action.from_database_id:
            raise ValueError(
                f'Action source {action.from_database_id!r} does not match current assignment '
                f'{state[query_index]!r} for query {action.query_id!r}'
            )

        next_state = list(state)
        next_state[query_index] = action.to_database_id
        return tuple(next_state)

    def _expand(self, node: Node, action: ReassignQuery) -> Node:
        child_state = self._apply_action(node.state, action)
        child = self._get_or_create_node(child_state)
        node.expanded_actions.add(action)
        node.children[action] = child
        return child

    def _select_best_child(self, node: Node, path_states: set[State]) -> Node | None:
        candidates = [child for child in node.children.values() if child.state not in path_states]
        if not candidates:
            return None

        best_score = -math.inf
        best_children: list[Node] = []
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

    def _evaluate_node(self, node: Node) -> float:
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

    def _compute_state_cost(self, state: State) -> float:
        return self._compute_state_cost_breakdown(state).total_cost

    def _compute_state_cost_breakdown(self, state: State) -> CostBreakdown:
        cached_breakdown = self.state_cost_breakdown_cache.get(state)
        if cached_breakdown is not None:
            return cached_breakdown

        if not self._is_valid_state(state):
            breakdown = CostBreakdown(
                total_cost=math.inf,
                latency_cost=math.inf,
                storage_cost=math.inf,
            )
            self.state_cost_cache[state] = math.inf
            self.state_cost_breakdown_cache[state] = breakdown
            self.state_reward_cache[state] = 0.0
            return breakdown

        latency_cost = self._compute_state_latency_cost(state)
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

    def _compute_state_latency_cost(self, state: State) -> float:
        total_cost = 0.0
        for query, database_id in zip(self.queries, state):
            database = self.database_by_id[database_id]
            latency = self._estimate_latency(query, database)
            total_cost += float(query.weight) * latency

        return total_cost

    def _compute_state_storage_cost(self, state: State) -> float:
        if self.estimate_storage_cost_fn is None:
            return 0.0

        table_ids_by_database_id: dict[DatabaseId, set[TableId]] = {}
        for query, database_id in zip(self.queries, state):
            table_ids = self.query_storage_table_ids_by_id.get(query.id, frozenset())
            if not table_ids:
                continue
            table_ids_by_database_id.setdefault(database_id, set()).update(table_ids)

        total_cost = 0.0
        for database_id, table_ids in table_ids_by_database_id.items():
            database = self.database_by_id[database_id]
            for table_id in table_ids:
                total_cost += self._estimate_storage_cost(table_id, database)

        return total_cost

    def _is_valid_state(self, state: State) -> bool:
        if len(state) != len(self.queries):
            return False

        for query, database_id in zip(self.queries, state):
            database = self.database_by_id.get(database_id)
            if database is None:
                return False
            if not self._can_execute(query, database):
                return False

        return True

    def _reward_for_cost(self, cost: float) -> float:
        if not math.isfinite(cost):
            return 0.0
        if self._baseline_cost == 0:
            return 1.0 if cost == 0 else 0.0
        return self._baseline_cost / max(cost, self.epsilon)

    def _cache_state_evaluation(self, node: Node, breakdown: CostBreakdown, reward: float):
        node.cost = breakdown.total_cost
        node.reward = reward
        self.state_cost_cache[node.state] = breakdown.total_cost
        self.state_cost_breakdown_cache[node.state] = breakdown
        self.state_reward_cache[node.state] = reward

    def _consider_best_state(self, state: State, breakdown: CostBreakdown, reward: float):
        if breakdown.total_cost < self._best_cost:
            self._best_state = state
            self._best_cost = breakdown.total_cost
            self._best_reward = reward
            self._best_latency_cost = breakdown.latency_cost
            self._best_storage_cost = breakdown.storage_cost

    def _state_to_assignment(self, state: State) -> Assignment:
        return {
            query.id: database_id
            for query, database_id in zip(self.queries, state)
        }


MCTS = MCTSOptimizer
