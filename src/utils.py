class ToCudaLoader:
    def __init__(self, loader):
        self.loader = loader
        self.batch_size = loader.batch_size

    def __iter__(self):
        for batch in self.loader:
            if isinstance(batch, (tuple, list)):
                yield [i.cuda(non_blocking=True) for i in batch]
            else:
                yield batch.cuda(non_blocking=True)

    def __len__(self):
        return len(self.loader)
