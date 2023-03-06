

class Cls(object):
    def __init__(self):
        self.a = 20

    def misete(self):
        print(self.a)

    def kansuukure(self):
        def kansuu():
            self.misete()
            return self.a

        return kansuu


if __name__ == "__main__":
    c = Cls()
    k = c.kansuukure()
    print(k())

    c.a = 30
    print(k())

    k = c.kansuukure()
    print(k())

    del c
    print(k())
