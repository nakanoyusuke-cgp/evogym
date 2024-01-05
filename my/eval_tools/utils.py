import numpy as np
import base64


def to_genecode(body: np.array) -> str:
    # 7 ^ 25通り
    # bodyを25桁の7進数とみなして, 10進数のintに変換
    # 9byte、bigエンディアン（実はどっちでもいい）に変換
    # base64にエンコード&デコード　文字列を返す
    str_n = ''
    for e in body.astype(np.int64).reshape(-1):
        str_n += str(e)
    return base64.b64encode(int(str_n, 7).to_bytes(9, "big")).decode()

def to_body(genecode: str) -> np.array:
    pass

def to_bit():
    pass


if __name__ == "__main__":
    for i in range(50):
        body =  np.random.randint(0, 7, (5,5))
        gc = to_genecode(body)
        print(gc)
