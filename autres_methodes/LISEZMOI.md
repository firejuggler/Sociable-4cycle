# Autres méthodes de construction de 4-cycles (hors Borho)

Ce dossier contient les prototypes écrits pendant l'exploration. Ils retrouvent des cycles connus du catalogue
par une autre voie que le théorème 1 de Borho, mais n'ont pas encore produit de cycle nouveau.

## Principe commun

On écrit chaque membre du cycle `n_i = G·m_i`, avec `pgcd(G, m_i) = 1` et `h = σ(G)/G`. La suite aliquote se
réduit alors aux cofacteurs :

    m_{i+1} = h·σ(m_i) − m_i

Les 74 4-cycles du catalogue qui n'ont pas la forme de Borho s'écrivent tous ainsi, avec quatre cofacteurs
`u_i` indépendants (voir `fit_catalogue.json`).

## Variante 1 : un grand premier par membre (`mitm.py`)

On pose `m_i = u_i·P_i`. Chaque étape est affine :

    P_{i+1} = (A_i·P_i + B_i)/u_{i+1},   avec A_i = h·σ(u_i) − u_i  et  B_i = h·σ(u_i)

En composant les quatre étapes, `P_1` est déterminé. On apparie au milieu `(u1, u2)` avec `(u3, u4)`.

    python3 mitm.py G U eps        # exemple : python3 mitm.py 4 2600 0.02

## Variante 2 : deux grands premiers dans un membre (`core.py`, `search3.py`)

On pose `m_1 = u_1·p·q`, et `m_i = u_i·P_i` pour i = 2, 3, 4. En faisant le tour du cycle :

    (C·p − D)(C·q − D) = N = D² + C·E

On trouve tous les `(p, q)` en énumérant les diviseurs `d ≡ −D (mod C)` de `N` (astuce de te Riele).
`core.coeffs(G, [u1,u2,u3,u4])` donne `(C, D, E)` et `core.solve(G, us)` donne les cycles, vérifiés exactement.

Ce qui compte, c'est que le C réduit soit petit. `search3.py` fait l'appariement au milieu sur une borne
inférieure de C, puis applique un crible de congruences qui élimine 90 à 99 % des candidats (dans `resid.py`,
qui lit `icoef.txt`). Les survivants sont factorisés avec PARI, puis avec yafu.

    python3 search3.py G U cap worker nworkers log NMIN tbudget [pari_sec]
    # exemple : python3 search3.py 4 3500 4000000 0 4 g4.log 1e14 600

Ce que search3.py demande :

- les paquets numpy, numba, sympy, cypari2 et cysignals ;
- `yafu_factor.py` (enveloppe Python de yafu), pour les nombres que PARI ne casse pas. Il cherche le binaire
  `yafu-linux-avx512ifma` dans le même dossier ; on peut aussi donner un autre chemin avec la variable
  d'environnement `YAFU` ;
- `NMIN` : taille minimale du cycle visé. La valeur 1e14 ignore la zone où le catalogue est déjà complet.

### Test de validation : le cycle n° 123 (1,8·10^19)

C'est le seul cycle hors Borho au-delà de 10^14. Avec G = 1, il a exactement cette forme :

    2^3·53·p·q  →  2^5·104831·P  →  2^5·17·P  →  2^3·11·P
    u = [424, 3354592, 544, 88],   C = 5107

    python3 -c "from core import solve; print(solve(1, [424, 3354592, 544, 88]))"

Cette commande retrouve le cycle 18232712294944907104 → … en quelques secondes. Trois cofacteurs sont
petits (au plus 544) et un seul est grand (environ 3,4·10^6). Les recherches lancées ici ne pouvaient pas
l'atteindre, parce qu'elles bornaient tous les u par la même limite (U ≤ 5000).

### Recherche asymétrique (`search_asym.py`)

Elle prend trois cofacteurs petits (u ≤ Us) et un grand (u dans [Ulo, Uhi]), placé en position `pos`
(1 à 4, et u1 porte p·q). La moitié A, (petit, grand), est parcourue à la volée ; la moitié B, (petit, petit),
est triée. Le crible et la factorisation sont les mêmes que dans search3.py.

    python3 search_asym.py G Us Ulo Uhi pos cap worker nworkers log NMIN tbudget [pari_sec] [eps]

Test de validation : sans connaître les u, elle retrouve le n° 123 en environ 4 min sur un cœur.

    python3 search_asym.py 1 600 3354000 3355000 2 4000000 0 1 t123.log 1e14 300

Le coût est d'environ 2·10^4 quadruplets par valeur du grand u (avec Us = 600). La phase de crible est la plus
longue : augmenter `tbudget` quand la plage [Ulo, Uhi] est large.

### Version GPU avec découpage automatique (`search_asym_gpu.py`, `asym_gpu.cu`)

Elle fait la même recherche que `search_asym.py`, avec trois différences :

- **Le noyau.** L'appariement est en C++/CUDA (`asym_gpu.cu`). La même source tourne sur la carte ou sur le CPU
  avec OpenMP, comme `apparie_gpu.cu`.
- **Un pré-filtre local dans le noyau.** Il vérifie que la chaîne est possible modulo 3^K, 5^K, 7^K et 2^K
  (avec K ≤ `--filtre`). Il ne rejette jamais un vrai cycle : testé sur les 109 cycles du catalogue de cette
  forme, aucun rejet. Il retire environ 95 % des candidats avant le crible Python. Le crible complet (`rho`) et
  la factorisation PARI s'appliquent ensuite aux survivants, répartis sur `--jobs` processus.
- **Un seuil de C fixe** (`--cmax`, 2e5 par défaut). La couverture est donc complète jusqu'à `cmax` sur toute la
  plage. Le programme découpe la plage du grand u en tranches : si une tranche donne plus de `--cap`
  candidats, il la coupe en deux et la refait ; si elle en donne peu, la tranche suivante est deux fois plus
  grande.

La reprise passe par `<out>.faits`, et Ctrl-C est sans danger. Pour compiler :

    nvcc -O3 -arch=sm_120 -Xcompiler -fPIC -shared -o libasym_gpu.so asym_gpu.cu
    # (version CPU : compilée automatiquement depuis asym_gpu.cu si l'on ne donne pas --gpu)

Test (cycle n° 123) :

    python3 search_asym_gpu.py 1 3354000 3355000 --pos 2 --out t123.log --gpu

Recherche :

    python3 search_asym_gpu.py 4 2000000 10000000 --pos 2 3 4 --gpu --jobs 8 --out g4_asym.log

Options :

| Option | Défaut | Rôle |
|---|---:|---|
| `--us` | 600 | borne des trois petits u |
| `--pos` | 2 | position(s) du grand u, de 1 à 4 |
| `--cmax` | 2e5 | seuil fixe du C réduit |
| `--cap` | 4e6 | candidats maximum par tranche (au-delà, la tranche est coupée) |
| `--tranche` | 20000 | taille de la première tranche, en nombre de grands u |
| `--filtre` | 128 | modules l^K du pré-filtre ; 0 le désactive |
| `--jobs` | nombre de cœurs | processus pour le crible et PARI |
| `--pari-sec` | 2 | temps maximal de PARI par N ; au-delà, N est noté `"hard": true` (pour yafu) |
| `--refaire` | — | ignorer `<out>.faits` |

Le noyau utilise 118 registres par fil (mesuré avec `-Xptxas -v`). Comme pour `apparie_gpu.cu`, ajouter
`-maxrregcount=64` peut accélérer : à mesurer sur la carte.

## Variante 3 : cycle mère → filles (`breed4.py`)

C'est une adaptation aux cycles de la règle de te Riele. Elle retrouve 63 cycles mères, mais ne produit
aucune fille : l'α à compenser dépasse 10^7, donc on attend environ τ(N)/α ≈ 0 solution.

## Résultats des recherches (aucun cycle nouveau)

| Recherche | Plage | Résultat |
|---|---|---|
| Variante 1 | G = 4, 8 ; u ≤ 2600 | retrouve les n° 4, 5, 6, 11 et 24 |
| Variante 2 | 16 valeurs de G, u ≤ 500–3000 (~26 M de N) | retrouve les n° 4, 5, 6 et 15 |
| Variante 2 + crible | G = 4 et G = 8, u ≤ 3500 | 0 |
| Variante 3 | 74 cycles mères hors Borho | 63 cycles mères retrouvés, 0 fille |
