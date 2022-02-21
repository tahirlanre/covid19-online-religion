class DoubleBertCollator(object):
    def __init__(self, source_collate_fn, target_collate_fn):
        self.source_collate_fn = source_collate_fn
        self.target_collate_fn = target_collate_fn

    def __call__(self, batch):
        domains = [b["domain"] for b in batch]
        if domains[0] == 0:
            return self.source_collate_fn(batch)
        else:
            return self.target_collate_fn(batch)