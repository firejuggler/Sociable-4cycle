"""yafu_factor(N, timeout=600, threads=1) -> dict {premier: exposant}, ou None (echec / timeout).
Usage CLI : python3 yafu_factor.py N [timeout] [threads]"""
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from math import prod

_ICI = os.path.dirname(os.path.abspath(__file__))
# binaire : variable d'environnement YAFU, sinon yafu-linux-avx512ifma dans ce dossier, sinon yafu/yafu
YAFU = os.environ.get("YAFU") or next(
    (c for c in (os.path.join(_ICI, "yafu-linux-avx512ifma"), os.path.join(_ICI, "yafu", "yafu")) if os.path.exists(c)),
    os.path.join(_ICI, "yafu-linux-avx512ifma"))


def yafu_factor(N, timeout=600, threads=1):
    N = int(N)
    if N < 2:
        return {}
    with tempfile.TemporaryDirectory() as d:          # yafu ecrit ses logs dans le repertoire courant
        try:
            out = subprocess.run([YAFU, "-threads", str(threads)], input=f"factor({N})\n",
                                 capture_output=True, text=True, timeout=timeout, cwd=d).stdout
        except subprocess.TimeoutExpired:
            return None
    if "***factors found***" not in out:
        return None
    bloc = out.split("***factors found***", 1)[1]
    f = Counter(int(x) for x in re.findall(r"^P\d+ = (\d+)", bloc, re.M))
    if not f or prod(p ** e for p, e in f.items()) != N:
        return None                                   # composite non casse (C..) ou sortie inattendue
    return dict(f)


if __name__ == "__main__":
    print(yafu_factor(sys.argv[1], *(int(a) for a in sys.argv[2:])))
