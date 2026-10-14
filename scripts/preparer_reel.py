"""Vrais enregistrements EIT : reconstruction par toutes les méthodes et données de l'application.

Pour chaque enregistrement (humains et animaux, données publiques d'EIDORS, appareil Goe-MF II à
16 électrodes) : lecture, vérification de l'ordre des mesures, détection des respirations, cycle
respiratoire moyen (16 phases, mesures moyennées phase par phase), puis reconstruction de chaque
phase par NOSER, le réseau de PulmoEIT (modèle figé) et quatre algorithmes d'EIDORS (GREIT,
Gauss-Newton en une étape et itératif, variation totale ; scripts/eidors_reel.m). Il n'y a pas de
vérité terrain : les méthodes sont comparées entre elles, à la physiologie attendue, à la stabilité
d'une respiration à l'autre et, pour la plèvre du porc, aux volumes injectés.

    python scripts/preparer_reel.py                          # tout (EIDORS : environ 1 h 30 en parallèle)
    python scripts/preparer_reel.py --sans-eidors            # réutilise les reconstructions EIDORS déjà faites
    python scripts/preparer_reel.py --recalculer porc_plevre # relance EIDORS pour un seul jeu
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))

from pulmoeit.fem import ForwardModel  # noqa: E402
from pulmoeit.grid import GRID, image_mask, pixel_centres  # noqa: E402
from pulmoeit.mesh import build_mesh  # noqa: E402
from pulmoeit.metrics import center_of_ventilation, global_inhomogeneity, pearson  # noqa: E402
from pulmoeit.peep import costa  # noqa: E402
from pulmoeit.protocol import adjacent_protocol  # noqa: E402
from pulmoeit.reel import (  # noqa: E402
    bruit_relatif,
    cycle_moyen,
    lire_get,
    respirations,
    signal_global,
    verifier_ordre,
    vers_protocole,
)
from pulmoeit.signals import features  # noqa: E402

DONNEES = Path.home() / "pulmoeit_reel"
TRAVAIL = RACINE / "data" / "reel"
SORTIE_WEB = RACINE / "demo" / "web" / "reel"
EIDORS = Path.home() / "outils_eit" / "eidors-v3.12"
IMAGE_OCTAVE = "gnuoctave/octave:9.2.0"
N_PHASES = 16
METHODES_EIDORS = ("GREIT", "Gauss-Newton une étape", "Gauss-Newton itératif", "Variation totale")
CONTENEURS = {"GREIT": 1, "Gauss-Newton une étape": 1, "Gauss-Newton itératif": 2, "Variation totale": 3}
EIDORS_URL = "https://eidors3d.sourceforge.net/data_contrib/"

PLEVRE = DONNEES / "jeux" / "gh_pleural_cavity" / "pig_data"
ENFANT = DONNEES / "peep" / "cg" / "DATA" / "STUDYNAME" / "SUBJECT_1" / "YYYYMMDD" / "Eit" / "Viasys"
# Chaque segment : (clé, titre, fichier, plage d'images 1-indexée incluse ou None, informations)
JEUX = [
    {
        "cle": "nouveau_ne_spontane",
        "titre": "Nouveau-né, respiration spontanée",
        "sujet": "humain",
        "sens": "gauche",
        "sens_documente": True,
        "source": "if-neonate-spontaneous",
        "citation": "I. Frerichs ; Heinrich et al., Intensive Care Med 2006;32:1392-1398",
        "description": "Nouveau-né de 10 jours, couché sur le ventre, tête tournée à gauche, respiration spontanée.",
        "segments": [
            ("spontane", "Respiration spontanée", RACINE / "data" / "reel" / "neonate" / "P04P-1016.get", None, {})
        ],
    },
    {
        "cle": "nouveau_ne_ventile",
        "titre": "Nouveau-né, ventilation mécanique",
        "sujet": "humain",
        "sens": "gauche",
        "sens_documente": False,
        "source": "if-neonate-ventilated",
        "citation": "I. Frerichs, données EIDORS « if-neonate-ventilated »",
        "description": "Nouveau-né sous ventilation mécanique (enregistrement de 58 s).",
        "segments": [
            ("ventile", "Ventilation mécanique", DONNEES / "jeux" / "if-neonate-ventilated" / "nm18.get", None, {})
        ],
    },
    {
        "cle": "enfant_sdra",
        "titre": "Enfant en SDRA, titration de la PEEP",
        "sujet": "humain",
        "sens": "gauche",
        "sens_documente": False,
        "source": "cg-2012-ards-recruitment",
        "citation": "GK Wolf, C Gómez-Laberge et al., Children's Hospital Boston 2009 (NCT00830284) ; "
        "C Gómez-Laberge et al., IEEE Trans Med Imag 2012;31:834-842",
        "description": "Enfant en SDRA, après une manœuvre de recrutement : PEEP abaissée de 20 à 14 cmH₂O par paliers.",
        "peep": True,
        "segments": [
            ("peep20", "PEEP 20 cmH₂O", ENFANT / "1001_d1.get", (1, 765), {"peep": 20.0, "delta_p": 6.0}),
            ("peep18", "PEEP 18 cmH₂O", ENFANT / "1001_d2.get", None, {"peep": 18.0, "delta_p": 6.0}),
            ("peep16", "PEEP 16 cmH₂O", ENFANT / "1001_d3.get", (368, 780), {"peep": 16.0, "delta_p": 5.0}),
            ("peep14", "PEEP 14 cmH₂O", ENFANT / "1001_d4.get", (411, 780), {"peep": 14.0, "delta_p": 4.0}),
        ],
    },
    {
        "cle": "adulte_sain",
        "titre": "Adulte sain",
        "sujet": "humain",
        "sens": "droite",
        "sens_documente": True,
        "source": "cg_normal_breathing_2006",
        "citation": "C Gómez-Laberge, Université d'Ottawa, 2005, données EIDORS « cg_normal_breathing_2006 »",
        "description": "Adulte de 25 ans en bonne santé, allongé sur le dos, ceinture au 5e espace intercostal.",
        "segments": [
            (
                "repos",
                "Au repos",
                DONNEES / "jeux" / "cg_normal_breathing_2006" / "cgomez-1pl" / "vivo150905_t1_001.get",
                None,
                {},
            ),
            (
                "effort",
                "Après un effort",
                DONNEES / "jeux" / "cg_normal_breathing_2006" / "cgomez-1pl" / "vivo150905_t1_004.get",
                None,
                {},
            ),
        ],
    },
    {
        "cle": "porc_plevre",
        "titre": "Porc : air puis liquide dans la plèvre",
        "sujet": "animal",
        # « électrodes numérotées dans le sens horaire, n° 1 en haut » : vu des pieds, l'électrode 5
        # est à gauche. Vérifié sur les données : l'air injecté à droite n'apparaît du côté droit de
        # l'image qu'avec ce sens, pour les six méthodes.
        "sens": "gauche",
        "sens_documente": True,
        "source": "gh_pleural_cavity",
        "citation": "G. Hahn, A. Just, T. Dudykevych, I. Frerichs et al., 2006, données EIDORS « gh_pleural_cavity »",
        "description": "Porc ventilé, couché sur le dos. Air puis liquide injectés dans la plèvre droite par paliers de 100 ml.",
        "reference": PLEVRE / "goev354005.get",
        "segments": [
            ("air0", "Air : 0 ml", PLEVRE / "goev354003_1-186.get", None, {"air": 0}),
            ("air300", "Air : 300 ml", PLEVRE / "goev354003_512-1560.get", None, {"air": 300}),
            ("liquide0", "Liquide : 0 ml", PLEVRE / "goev354008_2-270.get", None, {"liquide": 0}),
            ("liquide300", "Liquide : 300 ml", PLEVRE / "goev354008_947-1559.get", None, {"liquide": 300}),
        ],
        # changement de l'état du poumon par rapport à la référence, pour chaque volume
        "volumes": [
            ("air", 0, PLEVRE / "goev354003_1-186.get"),
            ("air", 100, PLEVRE / "goev354003_215-325.get"),
            ("air", 200, PLEVRE / "goev354003_361-512.get"),
            ("air", 300, PLEVRE / "goev354003_512-1560.get"),
            ("retrait", 300, PLEVRE / "goev354004_1-182.get"),
            ("retrait", 200, PLEVRE / "goev354004_218-364.get"),
            ("retrait", 100, PLEVRE / "goev354004_400-554.get"),
            ("retrait", 0, PLEVRE / "goev354004_702-1560.get"),
            ("liquide", 0, PLEVRE / "goev354008_2-270.get"),
            ("liquide", 100, PLEVRE / "goev354008_410-594.get"),
            ("liquide", 200, PLEVRE / "goev354008_727-856.get"),
            ("liquide", 300, PLEVRE / "goev354008_947-1559.get"),
        ],
    },
    {
        "cle": "porcelet_lavage",
        "titre": "Porcelet : sain, après lavage, après surfactant",
        "sujet": "animal",
        "sens": "droite",
        "sens_documente": True,
        "source": "Dargaville-ICM2010",
        "citation": "PA Dargaville, PC Rimensberger, I Frerichs, Intensive Care Med 2010;36:1953-1961",
        "description": "Porcelet ventilé : poumon sain, puis lésion par lavages répétés, puis surfactant. Premier palier (PEEP 0).",
        "premier_palier": True,
        "segments": [
            ("sain", "Poumon sain", DONNEES / "peep" / "dargaville" / "g10-c005.get", None, {}),
            ("lavage", "Après lavage (lésion)", DONNEES / "peep" / "dargaville" / "g10-c019.get", None, {}),
            ("surfactant", "Après surfactant", DONNEES / "peep" / "dargaville" / "g10-c029.get", None, {}),
        ],
    },
]


# ---------------------------------------------------------------- lecture


def charger(fichier: Path, plage, sens: str) -> np.ndarray:
    v = vers_protocole(lire_get(fichier), sens)
    return v if plage is None else v[plage[0] - 1 : plage[1]]


def premier_palier(v: np.ndarray, resp: list, seuil: float = 5.0, minimum: int = 4) -> list:
    """Respirations du premier palier d'un essai de PEEP : jusqu'au premier saut du niveau de fin
    d'expiration (plus de 5 fois sa variation habituelle d'une respiration à l'autre), le palier
    comptant au moins 4 respirations."""
    s = signal_global(v)
    fe = np.array([s[b] for b, _ in resp])
    d = np.diff(fe)
    bruit = 1.4826 * np.median(np.abs(d - np.median(d))) + 1e-12
    sauts = np.nonzero(np.abs(d) > seuil * bruit)[0]
    sauts = sauts[sauts >= minimum - 1]
    return resp[: sauts[0] + 1] if sauts.size else resp


def preparer_segments(reference_modele: np.ndarray) -> tuple[list[dict], list[dict]]:
    """Segments (cycle moyen) et images de changement d'état (plèvre)."""
    segments, changements = [], []
    for jeu in JEUX:
        for cle, titre, fichier, plage, infos in jeu["segments"]:
            v = charger(fichier, plage, jeu["sens"])
            resp = respirations(signal_global(v), 13)
            if jeu.get("premier_palier"):
                resp = premier_palier(v, resp)
            cycle = cycle_moyen(v, resp, N_PHASES)
            ordre = verifier_ordre(lire_get(fichier), reference_modele)
            segments.append(
                {
                    "jeu": jeu["cle"],
                    "cle": cle,
                    "titre": titre,
                    "infos": infos,
                    "respirations": len(resp),
                    "frequence": round(60 * 13 * (len(resp) - 1) / max(resp[-1][0] - resp[0][0], 1), 1),
                    "bruit": bruit_relatif(v, resp),
                    "ordre": ordre,
                    "cycle": cycle,
                    "signal": signal_global(cycle),
                    "par_respiration": np.stack([features(v[b], v[h])[0] for b, h in resp]),
                }
            )
            print(
                f"{jeu['cle']}/{cle} : {len(resp)} respirations, bruit {segments[-1]['bruit']:.3f}, ordre {ordre['decale']:.2f}",
                flush=True,
            )
        if "volumes" in jeu:
            ref = charger(jeu["reference"], None, jeu["sens"])
            fe_ref = np.mean([ref[b] for b, _ in respirations(signal_global(ref), 13)], axis=0)
            for etape, ml, fichier in jeu["volumes"]:
                v = charger(fichier, None, jeu["sens"])
                fe = np.mean([v[b] for b, _ in respirations(signal_global(v), 13)], axis=0)
                changements.append({"jeu": jeu["cle"], "etape": etape, "ml": ml, "vh": fe_ref, "vi": fe})
    return segments, changements


# ---------------------------------------------------------------- reconstructions


def _liste(tableaux: list) -> np.ndarray:
    sortie = np.empty(len(tableaux), dtype=object)
    for i, a in enumerate(tableaux):
        sortie[i] = a
    return sortie


def eidors(vh: np.ndarray, vi: np.ndarray, groupe: np.ndarray, bruit: np.ndarray) -> dict[str, np.ndarray]:
    """Lance les méthodes EIDORS en parallèle (plusieurs conteneurs pour les méthodes lentes, découpés
    par groupe d'images) ; retourne pour chaque méthode les valeurs (n_pixels, n_images)."""
    from scipy.io import loadmat, savemat

    injections, paires = adjacent_protocol()
    contenu = {
        "injections": injections,
        "paires": paires + 1,
        "vh": vh.T,
        "vi": vi.T,
        "groupe": groupe + 1,
        "bruit": bruit,
        "pixels": pixel_centres(GRID)[:, :, 0].reshape(-1, 2)[image_mask().ravel()],
    }
    for suffixe, n in (("", 160), ("_g", 96)):
        m = build_mesh(n_boundary=n)
        contenu |= {
            f"noeuds{suffixe}": m.nodes,
            f"elements{suffixe}": m.elements + 1,
            f"electrodes{suffixe}": _liste([np.unique(a) + 1 for a in m.electrode_edges]),
        }
    savemat(TRAVAIL / "entree_reel.mat", contenu)
    cache = Path.home() / "outils_eit" / "cache_eidors"
    cache.mkdir(exist_ok=True)
    lances = []
    t0 = time.perf_counter()
    for methode in METHODES_EIDORS:
        groupes = np.array_split(np.unique(groupe), CONTENEURS[methode])
        for k, gs in enumerate(groupes):
            colonnes = np.nonzero(np.isin(groupe, gs))[0]
            cle = "".join(c for c in methode.lower().replace(" ", "_") if c.isascii()) + f"_{k}"
            savemat(
                TRAVAIL / f"controle_{cle}.mat",
                {
                    "indice": METHODES_EIDORS.index(methode) + 1,
                    "debut": colonnes[0] + 1,
                    "fin": colonnes[-1] + 1,
                    "fichier": f"sortie_{cle}.mat",
                },
            )
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{EIDORS}:/eidors",
                "-v", f"{RACINE / 'scripts'}:/scripts:ro",
                "-v", f"{TRAVAIL}:/donnees",
                "-v", f"{cache}:/cache",
                "-v", f"{TRAVAIL / f'controle_{cle}.mat'}:/donnees/controle.mat:ro",
                IMAGE_OCTAVE, "octave-cli", "--no-gui", "--eval", "addpath('/scripts'); eidors_reel",
            ]  # fmt: skip
            journal = open(TRAVAIL / f"eidors_{cle}.log", "w")  # noqa: SIM115
            lances.append(
                (methode, cle, colonnes, subprocess.Popen(cmd, stdout=journal, stderr=subprocess.STDOUT), journal)
            )
    resultats: dict[str, np.ndarray] = {}
    reglages: dict[str, list] = {}
    for methode, cle, colonnes, p, journal in lances:
        p.wait()
        journal.close()
        if p.returncode != 0:
            raise RuntimeError(f"EIDORS {methode} : échec, voir {TRAVAIL / f'eidors_{cle}.log'}")
        s = loadmat(TRAVAIL / f"sortie_{cle}.mat", simplify_cells=True)
        bloc = resultats.setdefault(methode, np.zeros((int(image_mask().sum()), len(groupe))))
        bloc[:, colonnes] = np.atleast_2d(s["px"]).reshape(bloc.shape[0], -1)
        reglages.setdefault(methode, []).extend(np.atleast_1d(s["infos"]["reglage"]).tolist())
        print(f"EIDORS {methode} ({cle}) : {time.perf_counter() - t0:.0f} s", flush=True)
    (TRAVAIL / "reglages_eidors.json").write_text(json.dumps(reglages, ensure_ascii=False))
    np.savez_compressed(TRAVAIL / "eidors_images.npz", **{m: v for m, v in resultats.items()})
    return resultats


# ---------------------------------------------------------------- mesures


def couches(img: np.ndarray) -> np.ndarray:
    """Part (%) de la ventilation dans 4 couches de l'avant (sternum) vers l'arrière."""
    v = np.clip(img, 0, None) * image_mask()
    par = v.reshape(len(v), 4, GRID // 4, GRID).sum(axis=(2, 3))
    total = par.sum(axis=1, keepdims=True)
    return np.divide(100 * par, total, out=np.full_like(par, np.nan), where=total > 0)


def cotes(img: np.ndarray) -> np.ndarray:
    """Part (%) de la ventilation à droite du patient (gauche de l'image)."""
    v = np.clip(img, 0, None) * image_mask()
    d = v[:, :, : GRID // 2].sum(axis=(1, 2))
    return 100 * d / np.maximum(v.sum(axis=(1, 2)), 1e-30)


def octets(images: np.ndarray) -> str:
    """Séquence d'images -> octets (0 = pas d'air, 255 = haut de l'échelle de la méthode)."""
    pos = np.clip(images, 0, None) * image_mask()
    echelle = max(float(np.percentile(pos[:, image_mask()], 99.5)), 1e-12)
    return base64.b64encode((np.clip(pos / echelle, 0, 1) * 255).round().astype(np.uint8).tobytes()).decode()


def ecrire_geometrie() -> None:
    """Contour du thorax (masque de la grille) et position nominale des électrodes, pour l'affichage."""
    m = build_mesh(n_boundary=192)
    centres = np.array([m.nodes[a].reshape(-1, 2).mean(axis=0) for a in m.electrode_edges])
    SORTIE_WEB.mkdir(parents=True, exist_ok=True)
    (SORTIE_WEB / "geometrie.json").write_text(
        json.dumps(
            {
                "masque": image_mask().ravel().astype(int).tolist(),
                "electrodes": [[round((x + 1) / 2 * GRID, 3), round((1 - y) / 2 * GRID, 3)] for x, y in centres],
            }
        )
    )


def main() -> int:
    import onnxruntime as ort

    from pulmoeit.recon import LinearReconstructor

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sans-eidors", action="store_true", help="réutilise les images EIDORS du dernier calcul")
    parser.add_argument(
        "--recalculer",
        metavar="JEU",
        help="relance EIDORS pour ce jeu seulement (les autres images viennent du dernier calcul)",
    )
    args = parser.parse_args()
    TRAVAIL.mkdir(parents=True, exist_ok=True)
    m = build_mesh(n_boundary=160)
    reference_modele = ForwardModel(m).measure(np.ones(len(m.elements)))
    segments, changements = preparer_segments(reference_modele)

    # toutes les images à reconstruire : phases 1..15 de chaque cycle (la phase 0 est la référence)
    vh, vi, groupe, bruit = [], [], [], []
    for g, s in enumerate(segments):
        for k in range(1, N_PHASES):
            vh.append(s["cycle"][0]), vi.append(s["cycle"][k]), groupe.append(g)
        bruit.append(s["bruit"])
    g_changement = len(segments)
    for c in changements:
        vh.append(c["vh"]), vi.append(c["vi"]), groupe.append(g_changement)
    bruit.append(float(np.median([s["bruit"] for s in segments if s["jeu"] == "porc_plevre"])))
    vh, vi, groupe = np.stack(vh), np.stack(vi), np.array(groupe)
    x = features(vh, vi)
    images = {"NOSER (méthode des moniteurs)": LinearReconstructor.load(RACINE / "modeles" / "noser.npz")(x)}
    reseau = ort.InferenceSession(str(RACINE / "modeles" / "reseau.onnx"), providers=["CPUExecutionProvider"])
    images["Réseau PulmoEIT"] = reseau.run(None, {"dv": x.astype(np.float32)})[0].reshape(-1, GRID, GRID)
    if args.sans_eidors or args.recalculer:
        brut = np.load(TRAVAIL / "eidors_images.npz")
        eid = {m: brut[m].copy() for m in brut.files}
        if args.recalculer:
            gs = [g for g, s in enumerate(segments) if s["jeu"] == args.recalculer]
            if not gs:
                parser.error(f"jeu inconnu : {args.recalculer}")
            if args.recalculer == "porc_plevre":
                gs.append(g_changement)
            reglages = json.loads((TRAVAIL / "reglages_eidors.json").read_text())
            choisi = np.isin(groupe, gs)
            nouveaux = eidors(vh[choisi], vi[choisi], groupe[choisi], np.array(bruit))
            for methode in METHODES_EIDORS:
                eid[methode][:, choisi] = nouveaux[methode]
            # la variation totale a un réglage par groupe (dans l'ordre des groupes), les autres un seul
            tv = json.loads((TRAVAIL / "reglages_eidors.json").read_text())["Variation totale"]
            for g, valeur in zip(sorted(gs), tv, strict=True):
                reglages["Variation totale"][g] = valeur
            (TRAVAIL / "reglages_eidors.json").write_text(json.dumps(reglages, ensure_ascii=False))
            np.savez_compressed(TRAVAIL / "eidors_images.npz", **eid)
    else:
        eid = eidors(vh, vi, groupe, np.array(bruit))
    for methode in METHODES_EIDORS:
        img = np.zeros((len(groupe), GRID * GRID), np.float32)
        img[:, image_mask().ravel()] = -eid[methode].T  # ventilation = baisse de conductivité
        images[f"EIDORS {methode}"] = img.reshape(-1, GRID, GRID)
    noms = list(images)

    # ---- sortie web et rapport
    SORTIE_WEB.mkdir(parents=True, exist_ok=True)
    rapport: dict = {"methodes": noms, "jeux": {}}
    index = {"methodes": noms, "jeux": []}
    for jeu in JEUX:
        segs = [(g, s) for g, s in enumerate(segments) if s["jeu"] == jeu["cle"]]
        entree_index = {k: jeu[k] for k in ("cle", "titre", "sujet", "sens_documente", "citation", "description")}
        entree_index["url"] = EIDORS_URL + jeu["source"] + "/"
        entree_index["segments"] = [{"cle": s["cle"], "titre": s["titre"]} for _, s in segs]
        rapport["jeux"][jeu["cle"]] = {"segments": {}}
        for g, s in segs:
            sel = groupe == g
            mesures = {}
            panneaux = []
            for nom in noms:
                seq = np.concatenate([np.zeros((1, GRID, GRID), np.float32), images[nom][sel]])  # phase 0 = référence
                courant = seq[int(np.argmax(s["signal"]))]  # fin d'inspiration du cycle moyen
                mesures[nom] = {
                    "couches_pct": couches(courant[None])[0].round(1).tolist(),
                    "droite_pct": round(float(cotes(courant[None])[0]), 1),
                    "centre_pct": round(float(center_of_ventilation(courant[None])[0]), 1),
                    "gi": round(float(global_inhomogeneity(courant[None])[0]), 3),
                }
                panneaux.append({"nom": nom, "images": octets(seq)})
            # stabilité d'une respiration à l'autre (méthodes rapides en Python)
            par = s["par_respiration"]
            for nom, img in (
                ("NOSER (méthode des moniteurs)", LinearReconstructor.load(RACINE / "modeles" / "noser.npz")(par)),
                ("Réseau PulmoEIT", reseau.run(None, {"dv": par.astype(np.float32)})[0].reshape(-1, GRID, GRID)),
            ):
                moy = img.mean(axis=0, keepdims=True)
                mesures[nom]["stabilite"] = round(float(np.mean(pearson(img, np.repeat(moy, len(img), 0)))), 3)
            pic = int(np.argmax(s["signal"])) - 1  # fin d'inspiration (les images commencent à la phase 1)
            accord = {
                a: round(
                    float(
                        np.mean(
                            [
                                pearson(images[a][sel][pic : pic + 1], images[b][sel][pic : pic + 1])[0]
                                for b in noms
                                if b != a
                            ]
                        )
                    ),
                    3,
                )
                for a in noms
            }
            courbe = (s["signal"] - s["signal"].min()) / max(np.ptp(s["signal"]), 1e-30)
            donnees = {
                "titre": s["titre"],
                "infos": s["infos"],
                "respirations": s["respirations"],
                "frequence": s["frequence"],
                "images_par_seconde": round(N_PHASES / (60 / max(s["frequence"], 1)), 3),
                "n": N_PHASES,
                "courbe": [round(float(c), 4) for c in courbe],
                "panneaux": panneaux,
                "mesures": mesures,
            }
            (SORTIE_WEB / f"{jeu['cle']}_{s['cle']}.json").write_text(json.dumps(donnees, ensure_ascii=False))
            rapport["jeux"][jeu["cle"]]["segments"][s["cle"]] = {
                "respirations": s["respirations"],
                "frequence": s["frequence"],
                "bruit_relatif": round(s["bruit"], 4),
                "ordre_verifie": {k: round(v, 3) for k, v in s["ordre"].items()},
                "mesures": mesures,
                "accord_entre_methodes": accord,
            }
        if jeu.get("peep"):  # méthode de Costa sur les paliers documentés
            peeps = np.array([s["infos"]["peep"] for _, s in segs])
            dps = np.array([s["infos"]["delta_p"] for _, s in segs])
            res = {}
            for nom in noms:
                courants = np.stack([images[nom][groupe == g][int(np.argmax(s["signal"])) - 1] for g, s in segs])
                r = costa(courants, peeps, dps)
                res[nom] = {
                    "peep_proposee": round(r["peep_proposee"], 1),
                    "collapsus_pct": r["collapsus"].round(1).tolist(),
                    "surdistension_pct": r["surdistension"].round(1).tolist(),
                }
            entree_index["costa"] = {"peeps": peeps.tolist(), "methodes": res}
            rapport["jeux"][jeu["cle"]]["costa"] = entree_index["costa"]
        if "volumes" in jeu:  # plèvre : changement dans l'hémithorax droit en fonction du volume
            res = {}
            for nom in noms:
                img = images[nom][groupe == g_changement]
                droite = np.array(
                    [
                        np.clip(-im, 0, None)[:, : GRID // 2].sum() - np.clip(im, 0, None)[:, : GRID // 2].sum()
                        for im in img
                    ]
                )
                lignes = {}
                for etape in ("air", "retrait", "liquide"):
                    k = [i for i, c in enumerate(changements) if c["etape"] == etape]
                    ml = np.array([changements[i]["ml"] for i in k], float)
                    val = (
                        -droite[k] if etape != "liquide" else droite[k]
                    )  # air : impédance qui monte ; liquide : qui baisse
                    r = float(np.corrcoef(ml, val)[0, 1]) if np.ptp(val) > 0 else float("nan")
                    lignes[etape] = {
                        "ml": ml.tolist(),
                        "signal": (val / max(np.abs(val).max(), 1e-30)).round(3).tolist(),
                        "correlation": round(r, 3),
                    }
                res[nom] = lignes
                res[nom]["images"] = octets(img)
            entree_index["plevre"] = {
                "etapes": [{"etape": c["etape"], "ml": c["ml"]} for c in changements],
                "methodes": res,
            }
            rapport["jeux"][jeu["cle"]]["plevre"] = {
                n: {e: res[n][e]["correlation"] for e in ("air", "retrait", "liquide")} for n in noms
            }
        index["jeux"].append(entree_index)
    (SORTIE_WEB / "index.json").write_text(json.dumps(index, ensure_ascii=False))
    ecrire_geometrie()
    reglages = TRAVAIL / "reglages_eidors.json"
    rapport["reglages_eidors"] = json.loads(reglages.read_text()) if reglages.exists() else {}
    (RACINE / "reports" / "reel.json").write_text(json.dumps(rapport, indent=1, ensure_ascii=False))
    print("fini :", SORTIE_WEB)
    return 0


if __name__ == "__main__":
    sys.exit(main())
