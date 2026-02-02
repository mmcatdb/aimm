import dataclasses, json

# Add small epsilon to avoid division by zero
EPSILON = 1e-8

class JsonEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)
