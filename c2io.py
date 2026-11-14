"""
c2io.py — lecture des fichiers de paires c2_* quel que soit leur format (.txt, .gz, .gz « factorisation seule »)
=============================================================================================================
Format « factorisation seule » (FZ) : fichier gzip NORMAL (meme nom c2_X.txt.gz) dont le contenu commence
par la ligne MAGIC. Chaque ligne d'origine « n=factorisation » y est ecrite SANS n (« =factorisation »),
n etant le produit des facteurs ; toute autre ligne (en-tete, ligne vide, ligne dont le produit ne redonne
pas exactement n) est ecrite telle quelle, precedee de « ! ». La lecture reconstruit le contenu d'origine
OCTET POUR OCTET : les positions (reprise de borho4) sont les memes qu'avant la conversion.
Gain mesure : c2_131_6 : 142,4 Mo -> 61,9 Mo (gzip) -> 30,7 Mo (FZ).

API :
  ouvrir(path)        flux binaire du contenu d'origine (read, readline, iteration, tell, seek)
  ouvrir_texte(path)  idem en texte (utf-8)
  est_fz(path)        vrai si path est au format FZ
  encoder(lignes_src, fo)   ecrit le format FZ (fo : flux binaire, ex. gzip.open(..., "wb")) ;
                            renvoie (md5 du contenu d'origine, octets d'origine, lignes gardees telles quelles)
Ligne de commande :
  python3 c2io.py cat c2_131_6.txt.gz | head      (affiche le contenu d'origine, comme zcat)
  python3 c2io.py info c2_131_6.txt.gz            (format, taille)
"""
import gzip
import hashlib
import io
import re
import sys

import gmpy2

MAGIC = b"#C2FZ1\n"
_LIGNE = re.compile(rb"^(\d+)=([0-9][0-9^*]*)(\r?\n)?$")


def _produit(fact):
    """Produit des facteurs d'une factorisation « p^e*q*... » (bytes) -> mpz."""
    r = gmpy2.mpz(1)
    for t in fact.split(b"*"):
        i = t.find(b"^")
        if i < 0:
            r *= gmpy2.mpz(t.decode())
        else:
            r *= gmpy2.mpz(t[:i].decode()) ** int(t[i + 1:])
    return r


def _ligne_fz(l):
    """Ligne d'origine -> ligne FZ."""
    m = _LIGNE.match(l)
    if m and m.group(1)[:1] != b"0":
        try:
            if str(_produit(m.group(2))).encode() == m.group(1):
                return l[len(m.group(1)):]               # « =factorisation\n »
        except (ValueError, ZeroDivisionError):
            pass
    return b"!" + l


def _ligne_orig(l):
    """Ligne FZ -> ligne d'origine."""
    if l[:1] == b"=":
        fin = len(l)
        while fin and l[fin - 1] in (10, 13):
            fin -= 1
        return str(_produit(l[1:fin])).encode() + l
    if l[:1] == b"!":
        return l[1:]
    raise ValueError(f"ligne FZ invalide : {l[:60]!r}")


def encoder(lignes, fo):
    fo.write(MAGIC)
    h, n, telles = hashlib.md5(), 0, 0
    for l in lignes:
        h.update(l)
        n += len(l)
        e = _ligne_fz(l)
        if e[:1] == b"!":
            telles += 1
        fo.write(e)
    return h.hexdigest(), n, telles


class _FzRaw(io.RawIOBase):
    """Contenu d'origine d'un fichier FZ, reconstruit a la volee."""

    def __init__(self, path):
        self.path = path
        self._ouvrir()

    def _ouvrir(self):
        self._gz = io.BufferedReader(gzip.open(self.path, "rb"), 1 << 20)
        if self._gz.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"{self.path} : pas au format FZ")
        self._buf, self._i, self._pos = b"", 0, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def _remplir(self):
        morceaux, taille = [], 0
        for l in self._gz:
            o = _ligne_orig(l)
            morceaux.append(o)
            taille += len(o)
            if taille >= 1 << 20:
                break
        self._buf, self._i = b"".join(morceaux), 0
        return bool(self._buf)

    def readinto(self, b):
        if self._i >= len(self._buf) and not self._remplir():
            return 0
        k = min(len(b), len(self._buf) - self._i)
        b[:k] = self._buf[self._i:self._i + k]
        self._i += k
        self._pos += k
        return k

    def tell(self):
        return self._pos

    def seek(self, off, whence=0):
        if whence == 1:
            off += self._pos
        elif whence != 0:
            raise io.UnsupportedOperation("seek depuis la fin non supporte")
        if off < self._pos:
            self._gz.close()
            self._ouvrir()
        while self._pos < off:                          # avance en lisant (comme gzip)
            if self._i >= len(self._buf) and not self._remplir():
                break
            k = min(off - self._pos, len(self._buf) - self._i)
            self._i += k
            self._pos += k
        return self._pos

    def close(self):
        try:
            self._gz.close()
        finally:
            super().close()


def est_gzip(path):
    """Vrai si le fichier est compresse en gzip (reconnu a son contenu, pas a son nom)."""
    with open(path, "rb") as fh:
        return fh.read(2) == b"\x1f\x8b"


def est_fz(path):
    if not est_gzip(path):
        return False
    with gzip.open(path, "rb") as fh:
        return fh.read(len(MAGIC)) == MAGIC


def ouvrir(path, tampon=1 << 20):
    """Flux binaire du contenu d'origine (.txt, .gz classique ou .gz FZ, reconnus a leur contenu)."""
    path = str(path)
    if not est_gzip(path):
        return open(path, "rb")
    if est_fz(path):
        return io.BufferedReader(_FzRaw(path), tampon)
    return io.BufferedReader(gzip.open(path, "rb"), tampon)


def ouvrir_texte(path):
    return io.TextIOWrapper(ouvrir(path), encoding="utf-8", errors="replace", newline="")


def _main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("cat", "info"):
        sys.exit(__doc__)
    for p in sys.argv[2:]:
        if sys.argv[1] == "info":
            import os
            print(f"{p} : {'FZ (factorisation seule)' if est_fz(p) else ('gzip classique' if est_gzip(p) else 'texte')}, "
                  f"{os.path.getsize(p) / 1e6:,.1f} Mo")
        else:
            with ouvrir(p) as fh:
                try:
                    while True:
                        b = fh.read(1 << 20)
                        if not b:
                            break
                        sys.stdout.buffer.write(b)
                except BrokenPipeError:
                    return


if __name__ == "__main__":
    _main()
