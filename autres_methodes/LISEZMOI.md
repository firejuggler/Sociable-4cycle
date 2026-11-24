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
- `yafu_factor.py` (enveloppe Python de yafu), pour les nombres que PARI ne casse pas ; la ligne `YAFU = ...`
  donne le chemin du binaire ;
- `NMIN` : taille minimale du cycle visé. La valeur 1e14 ignore la zone où le catalogue est déjà complet.

### Test de validation : le cycle n° 123 (1,8·10^19)

C'est le seul cycle hors Borho au-delà de 10^14. Avec G = 1, il a exactement cette forme :

    2^3·53·p·q  →  2^5·104831·P  →  2^5·17·P  →  2^3·11·P
    u = [424, 3354592, 544, 88],   C = 5107

    python3 -c "from core import solve; print(solve(1, [424, 3354592, 544, 88]))"

Cette commande retrouve le cycle 18232712294944907104 → … en quelques secondes. Trois cofacteurs sont
petits (au plus 544) et un seul est grand (environ 3,4·10^6). Les recherches lancées ici ne pouvaient pas
l'atteindre, parce qu'elles bornaient tous les u par la même limite (U ≤ 5000).

Piste suggérée : une recherche asymétrique, avec trois u petits et un u très grand.

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
