class RefUpdateCache:
    def __init__(self):
        self.cache = {}

    def make_key(self, computed_refs, update_interval, alpha, fraction):
        key_refs = tuple((r[0],) for r in computed_refs)  # Only use start indices
        return (key_refs, update_interval, alpha, fraction)

    def store(self, computed_refs, update_interval, alpha, fraction, updated_refs, phase0):
        key = self.make_key(computed_refs, update_interval, alpha, fraction)
        self.cache[key] = {
            "updated_refs": updated_refs,
            "phase0": phase0
        }

    def retrieve(self, computed_refs, update_interval, alpha, fraction):
        key = self.make_key(computed_refs, update_interval, alpha, fraction)
        return self.cache.get(key, None)

    def clear(self):
        self.cache.clear()

ref_update_cache = RefUpdateCache()
