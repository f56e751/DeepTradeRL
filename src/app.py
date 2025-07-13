# src/app.py
# src/ 디렉터리를 PYTHONPATH에 포함한 상태로 실행해야 합니다.

# 패키지를 통해 Foo, Bar 클래스를 바로 가져올 수 있음
from mypackage import Foo, Bar

def main():
    f = Foo()
    print(f.hello())

    b = Bar()
    print(b.hello())

if __name__ == "__main__":
    main()
