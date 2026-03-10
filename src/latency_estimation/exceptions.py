from latency_estimation.common import NnOperator

class NeuralUnitNotFoundException(Exception):
    def __init__(self, operator: NnOperator):
        super().__init__(f'Neural unit not found for operator: {operator.key()}.')
        self.operator = operator
