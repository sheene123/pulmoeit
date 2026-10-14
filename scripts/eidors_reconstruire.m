% Reconstructions EIDORS pour scripts/essai_peep.py (GNU Octave, conteneur Docker).
%
% Même problème que NOSER dans PulmoEIT : maillage de reconstruction de référence, électrodes,
% protocole adjacent (injection d -> d+1, mesure U_m - U_m+1, 208 valeurs dans le même ordre),
% données différentielles normalisées. Pour chaque méthode, le réglage est choisi sur les essais
% de validation (erreur quadratique après ajustement du gain, comme NOSER), puis appliqué à tous.
% Sortie : valeurs aux centres des pixels de la grille 32 x 32 (élément le plus proche).

run('/eidors/eidors/startup.m');
eidors_cache('cache_path', '/cache');

v = load('/donnees/verite_val.mat');
entree = 'eidors_entree.mat';
if isfield(v, 'entree'), entree = strtrim(char(v.entree)); end
d = load(['/donnees/' entree]);
n_val = double(v.n_val);
limite = size(d.vh, 2);
if isfield(v, 'limite') && v.limite > 0, limite = double(v.limite); end
vh = d.vh(:, 1:limite);
vi = d.vi(:, 1:limite);

% ---------------------------------------------------------------- modèle direct
function fmdl = faire_fmdl(noeuds, elements, electrodes, injections, paires)
  fmdl = eidors_obj('fwd_model', 'pulmoeit');
  fmdl.nodes = double(noeuds);
  fmdl.elems = double(elements);
  fmdl.boundary = find_boundary(fmdl.elems);
  fmdl.gnd_node = 1;
  for i = 1:numel(electrodes)
    fmdl.electrode(i).nodes = double(electrodes{i}(:))';
    fmdl.electrode(i).z_contact = 0.01;
  end
  P = double(paires);
  n_el = rows(injections);
  for s = 1:n_el
    m = P(P(:, 1) == s, 2);
    M = zeros(numel(m), n_el);
    for r = 1:numel(m)
      M(r, m(r)) = 1;
      M(r, mod(m(r), n_el) + 1) = -1;
    end
    stim(s).stimulation = 'Amp';
    stim(s).stim_pattern = double(injections(s, :))';
    stim(s).meas_pattern = sparse(M);
  end
  fmdl.stimulation = stim;
  fmdl.solve = @fwd_solve_1st_order;
  fmdl.jacobian = @jacobian_adjoint;
  fmdl.system_mat = @system_mat_1st_order;
  fmdl.normalize_measurements = 1;
end
fmdl = faire_fmdl(d.noeuds, d.elements, d.electrodes, d.injections, d.paires);
fmdl_g = faire_fmdl(d.noeuds_g, d.elements_g, d.electrodes_g, d.injections, d.paires);

v0 = fwd_solve(mk_image(fmdl, 1));
fmdl_brut = fmdl_g; fmdl_brut.normalize_measurements = 0;  % solveurs itératifs : maillage grossier
v0_brut = fwd_solve(mk_image(fmdl_brut, 1)).meas;
% mêmes différences relatives (vi - vh) ./ vh, exprimées autour des tensions du modèle homogène :
% pour les solveurs itératifs, qui gèrent eux-mêmes leurs données
vh_modele = repmat(v0_brut, 1, size(vh, 2));
vi_modele = v0_brut .* (1 + (vi - vh) ./ vh);
c = corrcoef(v0.meas, mean(d.vh, 2));
printf('Cohérence du protocole : corrélation modèle EIDORS / mesures = %.3f\n', c(1, 2)); fflush(stdout);
if c(1, 2) < 0.8, error('protocole ou signe incohérent avec PulmoEIT'); end

% ---------------------------------------------------------------- tranche 3D pour GREIT
function f3 = extruder(f2, epaisseur, couches)
  % chaque triangle devient un prisme découpé en 3 tétraèdres (découpage conforme : sommets triés)
  n = rows(f2.nodes); z = linspace(0, epaisseur, couches);
  f3 = f2;
  f3.nodes = [repmat(f2.nodes, couches, 1), kron(z(:), ones(n, 1))];
  T = [];
  for c = 1:couches - 1
    for e = 1:rows(f2.elems)
      s = sort(f2.elems(e, :));
      b = s + (c - 1) * n; h = s + c * n;
      T = [T; b(1) b(2) b(3) h(3); b(1) b(2) h(2) h(3); b(1) h(1) h(2) h(3)];
    end
  end
  f3.elems = T;
  f3.boundary = find_boundary(T);
  for i = 1:numel(f2.electrode)
    en = f2.electrode(i).nodes(:);
    f3.electrode(i).nodes = reshape(en + (0:couches - 1) * n, 1, []);
  end
  f3.gnd_node = 1;
end
fmdl3 = extruder(fmdl, 0.1, 3);

% ---------------------------------------------------------------- méthodes
function imdl = base(fmdl, nom)
  imdl = eidors_obj('inv_model', nom);
  imdl.fwd_model = fmdl;
  imdl.reconst_type = 'difference';
  imdl.jacobian_bkgnd.value = 1;
end

function imdl = construire(fmdl, fmdl3, fmdl_brut, methode, hp)
  switch methode
    case 'GREIT'
      opt.imgsz = [32 32];
      opt.noise_figure = hp;
      opt.square_pixels = 1;
      opt.target_plane = 0.05;
      imdl = mk_GREIT_model(fmdl3, 0.25, [], opt);
    case 'Gauss-Newton une étape'
      imdl = base(fmdl, methode);
      imdl.solve = @inv_solve_diff_GN_one_step;
      imdl.RtR_prior = @prior_laplace;
      imdl.hyperparameter.value = hp;
    case 'Gauss-Newton itératif'
      imdl = base(fmdl_brut, methode);
      imdl.solve = @inv_solve_gn;
      imdl.RtR_prior = @prior_laplace;
      imdl.hyperparameter.value = hp;
      imdl.inv_solve_gn.max_iterations = 5;
      imdl.inv_solve_gn.verbose = 0;
    case 'Variation totale'
      imdl = base(fmdl_brut, methode);
      imdl.solve = @inv_solve_TV_pdipm;
      imdl.R_prior = @prior_TV;
      imdl.hyperparameter.value = hp;
      imdl.parameters.max_iterations = 10;
      imdl.parameters.term_tolerance = 1e-3;
      imdl.parameters.keep_iterations = 0;
  end
end

function px = vers_pixels(img, pixels)
  c = interp_mesh(img.fwd_model);
  n = rows(img.elem_data);
  if rows(c) ~= n  % GREIT : une valeur par pixel de sa grille, chaque pixel fait de plusieurs triangles
    if isfield(img.fwd_model, 'coarse2fine') && columns(img.fwd_model.coarse2fine) == n
      F = img.fwd_model.coarse2fine;
      c = full(F' * c(:, 1:2)) ./ full(sum(F, 1))';
    else
      k = rows(c) / n;
      c = squeeze(mean(reshape(c(:, 1:2)', 2, k, n), 2))';
    end
  end
  D = (pixels(:, 1) - c(:, 1)') .^ 2 + (pixels(:, 2) - c(:, 2)') .^ 2;
  [~, idx] = min(D, [], 2);
  px = img.elem_data(idx, :);
end

function px = reconstruire(imdl, vh, vi, pixels, par_image)
  if par_image
    px = [];
    for k = 1:size(vh, 2)
      px = [px, vers_pixels(inv_solve(imdl, vh(:, k), vi(:, k)), pixels)];
    end
  else
    px = vers_pixels(inv_solve(imdl, vh, vi), pixels);
  end
end

function e = erreur(px, verite)  % px : conductivité (baisse = air), verite : ventilation
  p = -px;
  g = sum(p(:) .* verite(:)) / max(sum(p(:) .^ 2), 1e-30);
  e = mean((g * p(:) - verite(:)) .^ 2);
end

methodes = {
  'GREIT',                  [0.25 0.5 1.0],          false, false;
  'Gauss-Newton une étape', [1e-3 1e-2 3e-2 1e-1 1], false, false;
  'Gauss-Newton itératif',  [1e-3 1e-2 1e-1],        true,  true;
  'Variation totale',       [1e-4 1e-3 1e-2],        true,  true;
};
pixels = double(d.pixels);
% une seule méthode par conteneur (lancés en parallèle par essai_peep.py) ; '' = toutes
seule = 0;
if isfield(v, 'indice'), seule = double(v.indice); end  % numéro (pas de nom : accents)
sortie = struct('nom', {}, 'hp', {}, 'scores', {}, 'pixels', {}, 'secondes_par_image', {});
for i = 1:rows(methodes)
  [nom, hps, par_image, unites_modele] = methodes{i, :};
  if seule > 0 && seule ~= i, continue; end
  % solveurs lents : réglage choisi sur une image de validation sur trois (tous les patients de validation)
  if par_image, idx = 1:3:n_val; else, idx = 1:n_val; end
  if isfield(v, 'hp_fixe'), hps = double(v.hp_fixe); end  % données réelles : réglage choisi en simulation
  if unites_modele, a = vh_modele; b = vi_modele; else, a = vh; b = vi; end
  scores = nan(size(hps));
  try
    for j = 1:numel(hps) * ~isfield(v, 'hp_fixe')
      t0 = tic;
      imdl = construire(fmdl, fmdl3, fmdl_brut, nom, hps(j));
      t1 = toc(t0);
      px = reconstruire(imdl, a(:, idx), b(:, idx), pixels, par_image);
      scores(j) = erreur(px, v.verite(:, idx));
      printf('%s, réglage %g : erreur validation %.3g (modèle %.1f s, %.2f s par image)\n', nom, hps(j), scores(j), t1, (toc(t0) - t1) / numel(idx));
      fflush(stdout);
    end
    [~, j] = min(scores);
    if isfield(v, 'hp_fixe'), j = 1; end
    imdl = construire(fmdl, fmdl3, fmdl_brut, nom, hps(j));
    t0 = tic;
    px = reconstruire(imdl, a, b, pixels, par_image);
    sec = toc(t0) / size(vh, 2);
    sortie(end + 1) = struct('nom', nom, 'hp', hps(j), 'scores', scores, 'pixels', px, 'secondes_par_image', sec);
    printf('%s : réglage retenu %g, %.3f s par image\n', nom, hps(j), sec); fflush(stdout);
  catch err
    printf('%s : ÉCHEC (%s)\n', nom, err.message); fflush(stdout);
  end
end
methodes = sortie;
fichier = '/donnees/eidors_sortie.mat';
if isfield(v, 'fichier'), fichier = ['/donnees/' strtrim(char(v.fichier))]; end
save('-v7', fichier, 'methodes');
