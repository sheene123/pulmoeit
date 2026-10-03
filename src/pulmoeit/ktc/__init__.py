"""PulmoEIT sur le Kuopio Tomography Challenge 2023 (KTC2023).

Benchmark public de reconstruction EIT : vraies mesures sur une cuve d'eau à 32 électrodes,
objets résistifs ou conducteurs, vérité terrain segmentée en trois classes et score officiel
(SSIM par classe). Même approche que PulmoEIT : une reconstruction linéaire physique, puis un
U-Net qui la corrige, entraîné sur des simulations faites avec le simulateur des organisateurs.

Données et code officiels : Räsänen et al., Zenodo, doi:10.5281/zenodo.8252370 (CC BY 4.0).
"""
