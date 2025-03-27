import json

class Hippocampus:
    def __init__(self):
        self.data = {}
    def write(self, values):
        self.data = values
        with open("C:/Users/milla/consciousness//organism/memoryhippocampus.json", "w") as f:
            json.dump(values, f)

    def read(self):
        return self.data