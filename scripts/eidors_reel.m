% Reconstructions EIDORS des vrais enregistrements (GNU Octave, conteneur Docker) pour
% scripts/preparer_reel.py.
%
% Même problème pour toutes les méthodes : maillage de reconstruction de PulmoEIT, électrodes,
% protocole adjacent (injection d -> d+1, mesure U_m - U_m+1, 208 valeurs dans le même ordre),
% données différentielles. Réglages sans simulation :
%   - GREIT : facteur de bruit (noise figure) 0,5, la valeur recommandée par EIDORS ;
%   - Gauss-Newton (une étape et itératif) : a priori laplacien, hyperparamètre choisi par EIDORS
%     pour le même facteur de bruit 0,5 (choose_noise_figure) ;
%   - variation totale (PDIPM) : principe de l'écart de Morozov, pour chaque enregistrement :
%     l'hyperparamètre retenu est celui dont l'écart relatif aux mesures est le plus proche du
%     bruit relatif mesuré d'une respiration à l'autre.
% Les méthodes itératives tournent sur un maillage plus grossier (1 440 éléments, toujours plus fin
% que la grille de 580 pixels).

run('/eidors/eidors/startup.m');
eidors_cache('cache_path', '/cache');
c = load('/donnees/controle.mat');
d = load('/donnees/entree_reel.mat');
indice = double(c.indice);  % numéro de méthode (les noms accentués ne passent pas d'un fichier .mat à l'autre)
noms = {'GREIT', 'Gauss-Newton une étape', 'Gauss-Newton itératif', 'Variation totale'};
methode = noms{indice};
colonnes = double(c.debut):double(c.fin);
vh = d.vh(:, colonnes);
vi = d.vi(:, colonnes);
groupe = double(d.groupe(colonnes));
pixels = double(d.pixels);

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

function f3 = extruder(f2, epaisseur, couches)
  % GREIT n'existe que pour des modèles 3D : le maillage 2D devient une fine tranche (prismes
  % découpés en 3 tétraèdres, sommets triés pour un découpage conforme).
  n = rows(f2.nodes); z = linspace(0, epaisseur, couches);
  f3 = f2;
  f3.nodes = [repmat(f2.nodes, couches, 1), kron(z(:), ones(n, 1))];
  T = zeros(3 * (couches - 1) * rows(f2.elems), 4); k = 0;
  for cc = 1:couches - 1
    for e = 1:rows(f2.elems)
      s = sort(f2.elems(e, :)); b = s + (cc - 1) * n; h = s + cc * n;
      T(k + 1:k + 3, :) = [b(1) b(2) b(3) h(3); b(1) b(2) h(2) h(3); b(1) h(1) h(2) h(3)]; k = k + 3;
    end
  end
  f3.elems = T;
  f3.boundary = find_boundary(T);
  for i = 1:numel(f2.electrode)
    f3.electrode(i).nodes = reshape(f2.electrode(i).nodes(:) + (0:couches - 1) * n, 1, []);
  end
  f3.gnd_node = 1;
end

function px = vers_pixels(img, pixels)
  cc = interp_mesh(img.fwd_model);
  n = rows(img.elem_data);
  if rows(cc) ~= n  % GREIT : une valeur par pixel de sa grille
    if isfield(img.fwd_model, 'coarse2fine') && columns(img.fwd_model.coarse2fine) == n
      F = img.fwd_model.coarse2fine;
      cc = full(F' * cc(:, 1:2)) ./ full(sum(F, 1))';
    else
      k = rows(cc) / n;
      cc = squeeze(mean(reshape(cc(:, 1:2)', 2, k, n), 2))';
    end
  end
  D = (pixels(:, 1) - cc(:, 1)') .^ 2 + (pixels(:, 2) - cc(:, 2)') .^ 2;
  [~, idx] = min(D, [], 2);
  px = img.elem_data(idx, :);
end

function imdl = gn_une_etape(fmdl, nf)
  imdl = eidors_obj('inv_model', 'GN');
  imdl.fwd_model = fmdl;
  imdl.reconst_type = 'difference';
  imdl.jacobian_bkgnd.value = 1;
  imdl.solve = @inv_solve_diff_GN_one_step;
  imdl.RtR_prior = @prior_laplace;
  cc = interp_mesh(fmdl);
  imdl.hyperparameter.func = @choose_noise_figure;
  imdl.hyperparameter.noise_figure = nf;
  imdl.hyperparameter.tgt_elems = find(sum(cc(:, 1:2) .^ 2, 2) < 0.15 ^ 2);
  imdl.hyperparameter.value = choose_noise_figure(imdl);
  imdl.hyperparameter = rmfield(imdl.hyperparameter, {'func'});
end

fmdl = faire_fmdl(d.noeuds, d.elements, d.electrodes, d.injections, d.paires);
fmdl_g = faire_fmdl(d.noeuds_g, d.elements_g, d.electrodes_g, d.injections, d.paires);
fmdl_brut = fmdl_g; fmdl_brut.normalize_measurements = 0;
v0 = fwd_solve(mk_image(fmdl_brut, 1)).meas;
% solveurs itératifs : mêmes différences relatives, autour des tensions du modèle homogène
vh_m = repmat(v0, 1, columns(vh));
vi_m = v0 .* (1 + (vi - vh) ./ vh);
t0 = tic;
infos = struct();
switch indice
  case 1
    opt.imgsz = [32 32]; opt.noise_figure = 0.5; opt.square_pixels = 1; opt.target_plane = 0.05;
    imdl = mk_GREIT_model(extruder(fmdl, 0.1, 3), 0.25, [], opt);
    px = vers_pixels(inv_solve(imdl, vh, vi), pixels);
    infos.reglage = 0.5;
  case 2
    imdl = gn_une_etape(fmdl, 0.5);
    px = vers_pixels(inv_solve(imdl, vh, vi), pixels);
    infos.reglage = imdl.hyperparameter.value;
  case 3
    lam = gn_une_etape(fmdl_brut, 0.5).hyperparameter.value;
    imdl = eidors_obj('inv_model', 'GNi');
    imdl.fwd_model = fmdl_brut; imdl.reconst_type = 'difference'; imdl.jacobian_bkgnd.value = 1;
    imdl.solve = @inv_solve_gn; imdl.RtR_prior = @prior_laplace; imdl.hyperparameter.value = lam;
    imdl.inv_solve_gn.max_iterations = 5; imdl.inv_solve_gn.verbose = 0;
    px = [];
    for k = 1:columns(vh)
      px = [px, vers_pixels(inv_solve(imdl, vh_m(:, k), vi_m(:, k)), pixels)];
    end
    infos.reglage = lam;
  case 4
    J = calc_jacobian(mk_image(fmdl_brut, 1));
    grille = [1e-6 1e-5 1e-4 1e-3];
    px = zeros(rows(pixels), columns(vh));
    reglages = [];
    for g = unique(groupe(:))'
      cols = find(groupe == g);
      dv = vi_m(:, cols) - vh_m(:, cols);
      [~, kref] = max(sum(dv .^ 2, 1));  % l'image de plus grande amplitude du groupe
      cible = double(d.bruit(g));
      ecarts = nan(size(grille));
      for j = 1:numel(grille)
        imdl = eidors_obj('inv_model', 'TV');
        imdl.fwd_model = fmdl_brut; imdl.reconst_type = 'difference'; imdl.jacobian_bkgnd.value = 1;
        imdl.solve = @inv_solve_TV_pdipm; imdl.R_prior = @prior_TV; imdl.hyperparameter.value = grille(j);
        imdl.parameters.max_iterations = 10; imdl.parameters.term_tolerance = 1e-3; imdl.parameters.keep_iterations = 0;
        x = inv_solve(imdl, vh_m(:, cols(kref)), vi_m(:, cols(kref))).elem_data;
        ecarts(j) = norm(J * x - dv(:, kref)) / norm(dv(:, kref));
      end
      [~, j] = min(abs(log(ecarts ./ cible)));
      imdl.hyperparameter.value = grille(j);
      for k = cols(:)'
        px(:, k) = vers_pixels(inv_solve(imdl, vh_m(:, k), vi_m(:, k)), pixels);
      end
      reglages(end + 1) = grille(j);
      printf('groupe %d : bruit relatif %.3f, écarts %s -> %g\n', g, cible, mat2str(ecarts, 3), grille(j)); fflush(stdout);
    end
    infos.reglage = reglages;
end
infos.secondes = toc(t0);
printf('%s : colonnes %d-%d, %.1f s\n', methode, c.debut, c.fin, infos.secondes); fflush(stdout);
save('-v7', ['/donnees/' strtrim(char(c.fichier))], 'px', 'infos');
