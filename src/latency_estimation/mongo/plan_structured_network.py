import torch
import torch.nn as nn
from typing import Dict, List, Any
from neural_units import GenericUnit
from feature_extractor import FeatureExtractor


class PlanStructuredNetwork(nn.Module):
    """
    Plan-structured neural network for MongoDB QPP.
    """

    def __init__(self, feature_extractor: FeatureExtractor,
                 hidden_dim: int = 128, num_layers: int = 3,
                 data_vec_dim: int = 32):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.data_vec_dim = data_vec_dim
        self.units = nn.ModuleDict()
        self.operator_info = {}

    # ------------------------------------------------------------------
    # Helpers for navigating both Classic and SBE plan trees
    # ------------------------------------------------------------------

    @staticmethod
    def get_children(node: Dict) -> List[Dict]:
        """Get child stages of a plan node (handles both inputStage/inputStages)."""
        children = []
        if "inputStage" in node:
            children.append(node["inputStage"])
        if "inputStages" in node:
            children.extend(node["inputStages"])
        return children

    @staticmethod
    def extract_plan_tree(explain_output: Dict) -> Dict:
        """
        Extract the plan tree from an explain output.
        Handles both explainVersion 1 (classic) and 2 (SBE / aggregation).
        """
        version = explain_output.get("explainVersion", "1")

        if version == "2":
            # SBE or aggregate: may have stages[] or queryPlanner.winningPlan.queryPlan
            if "stages" in explain_output:
                # Aggregate pipeline: get the cursor's queryPlan
                cursor_stage = explain_output["stages"][0]
                if "$cursor" in cursor_stage:
                    wp = cursor_stage["$cursor"]["queryPlanner"]["winningPlan"]
                    return wp.get("queryPlan", wp)
            wp = explain_output.get("queryPlanner", {}).get("winningPlan", {})
            return wp.get("queryPlan", wp)
        else:
            # Classic engine
            wp = explain_output.get("queryPlanner", {}).get("winningPlan", {})
            return wp

    @staticmethod
    def extract_execution_tree(explain_output: Dict) -> Dict:
        """
        Extract the execution stats tree from an explain output (for training).
        """
        version = explain_output.get("explainVersion", "1")

        if version == "2":
            if "stages" in explain_output:
                cursor_stage = explain_output["stages"][0]
                if "$cursor" in cursor_stage:
                    return cursor_stage["$cursor"]["executionStats"]["executionStages"]
            return explain_output.get("executionStats", {}).get("executionStages", {})
        else:
            return explain_output.get("executionStats", {}).get("executionStages", {})

    @staticmethod
    def get_collection_from_explain(explain_output: Dict) -> str:
        """Extract collection name from explain output."""
        version = explain_output.get("explainVersion", "1")
        if version == "2":
            if "stages" in explain_output:
                cursor_stage = explain_output["stages"][0]
                if "$cursor" in cursor_stage:
                    ns = cursor_stage["$cursor"]["queryPlanner"].get("namespace", "")
                    return ns.split(".")[-1] if "." in ns else ns
        ns = explain_output.get("queryPlanner", {}).get("namespace", "")
        return ns.split(".")[-1] if "." in ns else ns

    def _normalize_stage(self, stage: str) -> str:
        """Normalize stage name for unit lookup."""
        # Keep all stages distinct - no grouping needed for MongoDB
        return stage

    # ------------------------------------------------------------------
    # Unit creation and management
    # ------------------------------------------------------------------

    def initialize_units_from_plans(self, plans: List[Dict]):
        """
        Scan all plan trees to discover operator types and create neural units.
        plans: list of explain outputs (full explain dicts).
        """
        type_children_pairs = set()

        def collect(node):
            stage = self._normalize_stage(node.get("stage", "UNKNOWN"))
            children = self.get_children(node)
            num_children = len(children)
            type_children_pairs.add((stage, num_children))
            for child in children:
                collect(child)

        for plan in plans:
            tree = self.extract_plan_tree(plan)
            collect(tree)

        print(f"\nFound {len(type_children_pairs)} operator/children combinations:")
        for stage, nc in sorted(type_children_pairs):
            feat_dim = self.feature_extractor.get_feature_dim(stage)
            print(f"  {stage} (children={nc}): feature_dim={feat_dim}")
            op_key = f"{stage}_{nc}"
            self.operator_info[op_key] = {
                "node_type": stage,
                "feature_dim": feat_dim,
                "num_children": nc,
            }
            self._create_unit(stage, feat_dim, nc)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"\nInitialized {len(self.units)} neural units, {total_params:,} parameters")

    def initialize_units_from_operator_info(self, operator_info: Dict):
        """Recreate units from saved operator info (for loading checkpoints)."""
        for op_key, info in operator_info.items():
            self._create_unit(info["node_type"], info["feature_dim"], info["num_children"])
        self.operator_info = dict(operator_info)

    def _create_unit(self, stage: str, feature_dim: int, num_children: int = 0):
        op_key = f"{stage}_{num_children}"
        if op_key in self.units:
            return self.units[op_key]
        unit = GenericUnit(
            input_dim=feature_dim,
            num_children=num_children,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            data_vec_dim=self.data_vec_dim,
        )
        self.units[op_key] = unit
        return unit

    def get_unit(self, stage: str, num_children: int) -> nn.Module:
        op_key = f"{stage}_{num_children}"
        if op_key not in self.units:
            # Dynamic creation for unseen combinations
            feat_dim = self.feature_extractor.get_feature_dim(stage)
            return self._create_unit(stage, feat_dim, num_children)
        return self.units[op_key]

    def get_operator_info(self) -> Dict:
        return dict(self.operator_info)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def process_node(self, node: Dict, collection_name: str,
                     cache: Dict = None) -> torch.Tensor:
        """
        Recursively process a plan tree node bottom-up.

        Returns: tensor [1, 1 + data_vec_dim]  (latency + data vector)
        """
        if cache is None:
            cache = {}

        node_id = id(node)
        if node_id in cache:
            return cache[node_id]

        stage = self._normalize_stage(node.get("stage", "UNKNOWN"))
        children = self.get_children(node)
        children_outputs = []
        for child in children:
            child_out = self.process_node(child, collection_name, cache)
            children_outputs.append(child_out)

        num_children = len(children_outputs)
        unit = self.get_unit(stage, num_children)

        # Extract features
        features = self.feature_extractor.extract_features(node, collection_name)
        device = next(self.parameters()).device
        feat_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)

        if children_outputs:
            children_concat = torch.cat(children_outputs, dim=1)
            unit_input = torch.cat([feat_tensor, children_concat], dim=1)
        else:
            unit_input = feat_tensor

        output = unit(unit_input)
        cache[node_id] = output
        return output

    def forward(self, plan_tree: Dict, collection_name: str) -> torch.Tensor:
        """
        Predict latency for a query plan.

        Args:
            plan_tree: the plan tree root node (already extracted from explain)
            collection_name: name of the primary collection

        Returns: scalar predicted latency (ms)
        """
        cache = {}
        output = self.process_node(plan_tree, collection_name, cache)
        return output[:, 0]  # latency

    def get_all_node_predictions(self, plan_tree: Dict,
                                 collection_name: str) -> Dict:
        """Get predictions for all nodes (used in loss computation)."""
        cache = {}
        self.process_node(plan_tree, collection_name, cache)
        return cache
