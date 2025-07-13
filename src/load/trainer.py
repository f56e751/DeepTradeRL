from ..mypackage import Foo, Bar

def train():
    f = Foo()
    print(f.hello())
    b = Bar()
    print(b.hello())

if __name__ == "__main__":
    train()
