import categories


class others:
    def __init__(self):
        self.main_others_count = 0
        self.sub_others_counts = {
            categories.COMP: 0,
            categories.REC: 0,
            categories.SCI: 0,
            categories.POLITICS: 0,
            categories.REL: 0,
            categories.MISC: 0
        }

    def set_main_others_count(self, count):
        self.main_others_count = count
