from manim import *
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture

config.max_files_cached = 1000


config.pixel_height = 2160
config.pixel_width  = 3840
config.frame_rate = 60

# ─── colour helpers ─────────────────────────────────────────────────────────────

def manim_to_rgb(col) -> np.ndarray:
    return np.array(ManimColor(col).to_rgb(), dtype=float)

def rgb_to_manim(rgb: np.ndarray):
    return ManimColor.from_rgb(tuple(float(v) for v in np.clip(rgb, 0, 1)))

def blend_rgb(weights: np.ndarray, rgbs: np.ndarray) -> np.ndarray:
    return np.clip((weights[:, None] * rgbs).sum(axis=0), 0, 1)


# ─── clustering ─────────────────────────────────────────────────────────────────

def closest_drone_colors(centers_xy, drone_positions, drone_colors):
    """Map each cluster centroid to the colour of the nearest drone."""
    drone_xy = np.array([p[:2] for p in drone_positions])
    return [drone_colors[int(np.argmin(np.linalg.norm(drone_xy - c, axis=1)))]
            for c in centers_xy]


def run_kmeans(pts_xy, drone_positions, drone_colors, k=3):
    km      = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels  = km.fit_predict(pts_xy)
    colors  = closest_drone_colors(km.cluster_centers_, drone_positions, drone_colors)
    return [colors[lbl] for lbl in labels], km.cluster_centers_, colors


def run_dpgmm(pts_xy, drone_positions, drone_colors, k=3):
    bgm = BayesianGaussianMixture(
        n_components=k,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=1e-2,
        max_iter=300,
        random_state=42,
    )
    bgm.fit(pts_xy)
    resp    = bgm.predict_proba(pts_xy)            # (N, k)
    colors  = closest_drone_colors(bgm.means_, drone_positions, drone_colors)
    rgbs    = np.array([manim_to_rgb(c) for c in colors])   # (k, 3)
    dot_colors = [rgb_to_manim(blend_rgb(r, rgbs)) for r in resp]
    # Return responsibilities (resp) so we can scale based on probability
    return dot_colors, bgm.means_, bgm.covariances_, colors, resp


# ─── scene ──────────────────────────────────────────────────────────────────────

class DroneScene(MovingCameraScene):

    # ── geometry helpers ────────────────────────────────────────────────────────

    def sample_perimeter(self, vmobj: VMobject, step: float = 0.18) -> np.ndarray:
        pts = vmobj.get_points()
        if len(pts) < 2:
            return pts.copy()
        seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cumlen   = np.concatenate([[0], np.cumsum(seg_lens)])
        total    = cumlen[-1]
        positions = np.arange(0, total, step)
        if len(positions) == 0:
            return pts[:1]
        positions[-1] = min(positions[-1], total)
        out = []
        for t in positions:
            idx = min(np.searchsorted(cumlen, t, side="right") - 1,
                      len(seg_lens) - 1)
            sl = seg_lens[idx]
            a  = (t - cumlen[idx]) / sl if sl > 0 else 0.0
            out.append(pts[idx] + a * (pts[idx + 1] - pts[idx]))
        return np.array(out)

    def points_inside_area(self, points, area, margin=0.06):
        v  = area.get_vertices()
        x0, x1 = v[:, 0].min() + margin, v[:, 0].max() - margin
        y0, y1 = v[:, 1].min() + margin, v[:, 1].max() - margin
        mask = ((points[:, 0] >= x0) & (points[:, 0] <= x1) &
                (points[:, 1] >= y0) & (points[:, 1] <= y1))
        return points[mask]

    def get_gmm_ellipses(self, means, covars, colors, num_stdev=2.0):
        """Builds accurately scaled and rotated ellipses from GMM covariances."""
        ellipses = VGroup()
        for mean, cov, col in zip(means, covars, colors):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            
            theta = np.arctan2(*vecs[:, 0][::-1])
            width, height = 2 * num_stdev * np.sqrt(vals)
            
            ell = Ellipse(
                width=width, height=height,
                fill_color=col, fill_opacity=0.2,
                stroke_color=col, stroke_width=2.5,
                z_index=2.5
            )
            ell.rotate(theta)
            ell.move_to(np.array([mean[0], mean[1], 0]))
            ellipses.add(ell)
        return ellipses

    def get_info_gain_scales(self, pts_xy, drone_positions, fov_r, area, sq_size=0.4, res=10):
        """Calculates information gain by sampling 'unknown' area in squares around dots."""
        gains = []
        v = area.get_vertices()
        x0, x1 = v[:, 0].min(), v[:, 0].max()
        y0, y1 = v[:, 1].min(), v[:, 1].max()
        drone_ctrs = np.array([p[:2] for p in drone_positions])

        for pt in pts_xy:
            xs = np.linspace(pt[0] - sq_size/2, pt[0] + sq_size/2, res)
            ys = np.linspace(pt[1] - sq_size/2, pt[1] + sq_size/2, res)
            xv, yv = np.meshgrid(xs, ys)
            grid = np.c_[xv.ravel(), yv.ravel()]

            in_area = (grid[:, 0] >= x0) & (grid[:, 0] <= x1) & \
                      (grid[:, 1] >= y0) & (grid[:, 1] <= y1)
            valid = grid[in_area]

            if len(valid) == 0:
                gains.append(0.0)
                continue

            dists = np.linalg.norm(valid[:, None, :] - drone_ctrs[None, :, :], axis=2)
            in_fov = np.any(dists <= fov_r, axis=1)

            unknown_frac = np.sum(~in_fov) / len(grid) 
            gains.append(unknown_frac)

        gains = np.array(gains)
        g_min, g_max = gains.min(), gains.max()
        if g_max > g_min:
            norm_gains = 0.1 + 1.9 * (gains - g_min) / (g_max - g_min)
        else:
            norm_gains = np.ones_like(gains)

        return norm_gains

    # ── visual factories ────────────────────────────────────────────────────────

    def drone_fov_aoi(self, fov_circle, area):
        return Intersection(
            fov_circle, area,
            fill_color=WHITE, fill_opacity=0.30,
            stroke_width=0, z_index=2,
        )

    def build_fov_slice(self, pos, fov_r, area):
        """The FoV ∩ area shape for a single drone at a given world position.
        Used to grow a cumulative 'known area' union as a drone moves, rather
        than recomputing coverage from scratch (which would forget areas the
        drone previously saw before it moved)."""
        circle = Circle(radius=fov_r).move_to(pos)
        return self.drone_fov_aoi(circle, area)

    def normalize_gains(self, gains):
        """Stretch raw (0-1) info gains to the same 0.1-2.0 display range used
        for dot scaling elsewhere in the scene."""
        g_min, g_max = gains.min(), gains.max()
        if g_max > g_min:
            return 0.1 + 1.9 * (gains - g_min) / (g_max - g_min)
        return np.ones_like(gains)

    def _frontier_step(self, cfg):
        """Pure computation for one exploration round on a single panel — no
        self.play calls. Assigns ALL THREE drones a target simultaneously
        (not one at a time), then unions all their new FoV slices into the
        known area in one go. Reads/writes `cfg['state']`. Returns a dict
        describing what should be animated this round, or {'done': True}
        once nothing meaningful remains for this panel."""
        st = cfg['state']
        area, fov_r = cfg['area'], cfg['fov_r']
        zoom_center, scale, target_cx, extra_shift = (
            cfg['zoom_center'], cfg['scale'], cfg['target_cx'], cfg['extra_shift']
        )
        mode, colors = cfg['mode'], cfg['colors']
        base_radius, step, margin, gain_threshold = (
            cfg['base_radius'], cfg['step'], cfg['margin'], cfg['gain_threshold']
        )

        def to_panel_space(mobj):
            return (
                mobj.copy()
                .scale(scale, about_point=zoom_center)
                .shift(np.array([target_cx - zoom_center[0], -zoom_center[1], 0]))
                .shift(extra_shift)
            )

        raw_pts = self.sample_perimeter(st['known_world'], step=step)
        f_pts   = self.points_inside_area(raw_pts, area, margin=margin)
        if len(f_pts) == 0:
            return {'done': True}
        pts_xy = f_pts[:, :2]

        gains = self.raw_info_gain(pts_xy, st['visited_positions'], fov_r, area)
        if gains.max() < gain_threshold:
            return {'done': True}
        dot_scales = self.normalize_gains(gains)

        dists_all = np.stack([
            np.linalg.norm(pts_xy - cp[:2], axis=1) for cp in st['current_positions']
        ], axis=1)  # (N, 3)

        rgbs = np.array([manim_to_rgb(c) for c in colors])
        n_drones = len(st['current_positions'])

        if mode == "hard":
            nearest = np.argmin(dists_all, axis=1)
            dot_colors = [colors[n] for n in nearest]

            # How close each point sits to the boundary between two drones'
            # clusters: 1.0 = exactly on the boundary (nearest and second-
            # nearest drone equidistant), 0.0 = deep inside one drone's
            # territory. Used to boost contested/ambiguous boundary points.
            sorted_d = np.sort(dists_all, axis=1)
            d0, d1 = sorted_d[:, 0], sorted_d[:, 1]
            edge_score = np.where(d1 > 1e-9, 1.0 - (d1 - d0) / d1, 1.0)
            edge_boost = cfg.get('edge_boost', 0.0)
            weighted_gains = gains * (1.0 + edge_boost * edge_score)

            # Each drone's best point = highest (boosted) gain among points
            # nearest to it
            assignments = {}
            for d in range(n_drones):
                own = np.where(nearest == d)[0]
                if len(own) == 0:
                    continue
                assignments[d] = own[int(np.argmax(weighted_gains[own]))]
        else:
            weights = np.exp(-(dists_all ** 2) / (2 * fov_r ** 2))
            weights = weights / weights.sum(axis=1, keepdims=True)
            dot_colors = [rgb_to_manim(blend_rgb(w, rgbs)) for w in weights]
            # Greedy conflict-free matching: repeatedly grab the single best
            # remaining (drone, point) pair by score = weight x gain, so no
            # two drones are ever sent to the same point.
            score_matrix = weights * gains[:, None]  # (N, n_drones)
            assigned_points = set()
            remaining_drones = set(range(n_drones))
            assignments = {}
            for _ in range(n_drones):
                if not remaining_drones:
                    break
                best_val, best_pair = -1.0, None
                for d in remaining_drones:
                    col = score_matrix[:, d].copy()
                    if assigned_points:
                        col[list(assigned_points)] = -1.0
                    idx = int(np.argmax(col))
                    if col[idx] > best_val:
                        best_val, best_pair = col[idx], (d, idx)
                if best_pair is None or best_val <= 0:
                    break
                d, idx = best_pair
                assignments[d] = idx
                assigned_points.add(idx)
                remaining_drones.discard(d)

        if not assignments:
            return {'done': True}

        new_dots_world = VGroup(*[
            Dot(pt, radius=base_radius * s, color=c, z_index=3)
            for pt, s, c in zip(f_pts, dot_scales, dot_colors)
        ])
        new_dots_panel = to_panel_space(new_dots_world)

        max_step = cfg.get('max_step')

        def world_to_panel(pos):
            scaled = zoom_center + (np.array(pos, dtype=float) - zoom_center) * scale
            return scaled + np.array([target_cx - zoom_center[0], -zoom_center[1], 0]) + extra_shift

        # Union in every assigned drone's new FoV slice at once
        new_known_world = st['known_world']
        drone_targets = {}
        for d, best_i in assignments.items():
            raw_target = f_pts[best_i]
            cur_pos = st['current_positions'][d]
            delta = raw_target - cur_pos
            dist = float(np.linalg.norm(delta[:2]))
            if max_step is not None and dist > max_step and dist > 1e-9:
                # Only advance up to max_step toward the chosen point this round —
                # it'll keep closing the gap on subsequent rounds if still needed.
                target_pos_world = cur_pos + delta * (max_step / dist)
            else:
                target_pos_world = raw_target
            drone_targets[d] = {
                'target_pos_world': target_pos_world,
                'target_pos_panel': world_to_panel(target_pos_world),
            }
            new_slice = self.build_fov_slice(target_pos_world, fov_r, area)
            new_known_world = Union(
                new_known_world, new_slice,
                fill_color=WHITE, fill_opacity=0.25, stroke_width=0, z_index=2,
            )
        new_area_panel = to_panel_space(new_known_world)

        return {
            'done': False,
            'new_dots_panel': new_dots_panel,
            'drone_targets': drone_targets,
            'new_area_panel': new_area_panel,
            'new_known_world': new_known_world,
        }

    def coverage_percentage(self, visited_positions, fov_r, area, res=50):
        """Percentage of `area` currently within `fov_r` of any visited
        position, estimated via a grid sample."""
        v = area.get_vertices()
        x0, x1 = v[:, 0].min(), v[:, 0].max()
        y0, y1 = v[:, 1].min(), v[:, 1].max()
        xs = np.linspace(x0, x1, res)
        ys = np.linspace(y0, y1, res)
        xv, yv = np.meshgrid(xs, ys)
        grid = np.c_[xv.ravel(), yv.ravel()]
        ctrs = np.array([p[:2] for p in visited_positions])
        dists = np.linalg.norm(grid[:, None, :] - ctrs[None, :, :], axis=2)
        covered = np.any(dists <= fov_r, axis=1)
        return 100.0 * covered.mean()

    def _make_pct_text(self, frac, anchor):
        return Text(
            f"Explored: {frac:.0f}%", font_size=28, color=WHITE
        ).next_to(anchor, DOWN, buff=0.3)

    def is_point_covered(self, point, visited_positions, fov_r):
        """Whether `point` (world coords) currently sits within `fov_r` of
        any visited position."""
        pt_xy = np.array(point[:2], dtype=float)
        for vp in visited_positions:
            if np.linalg.norm(pt_xy - np.array(vp[:2], dtype=float)) <= fov_r:
                return True
        return False

    def _make_status_text(self, found, anchor):
        label = "Found" if found else "Lost"
        color = GREEN if found else RED
        return Text(
            f"Luigi Status : {label}", font_size=28, color=color
        ).next_to(anchor, DOWN, buff=0.2)

    def explore_synced(self, panel_configs, max_moves=6):
        """Drive several panels' dynamic-frontier exploration in lockstep:
        each round, every still-active panel's frontier dots are swapped in
        one combined self.play, then ALL THREE drones of every still-active
        panel move (+ grow their FoV) in a second combined self.play — so
        every drone on both sides of the split screen visibly moves at the
        same time. A panel that finishes early (nothing meaningful left to
        explore) simply drops out while the others continue. `panel_configs`
        is a list of dicts, each with keys: panel, area, fov_r, zoom_center,
        scale, target_cx, extra_shift, mode, colors, drone_positions_start,
        known_world (the starting coverage shape). Returns a dict of final
        known_world per config id."""
        for i, cfg in enumerate(panel_configs):
            cfg.setdefault('id', i)
            cfg.setdefault('base_radius', 0.055)
            cfg.setdefault('step', 0.3)
            cfg.setdefault('margin', 0.05)
            cfg.setdefault('gain_threshold', 0.05)
            cfg['state'] = {
                'known_world': cfg['known_world'],
                'current_positions': list(cfg['drone_positions_start']),
                'visited_positions': list(cfg['drone_positions_start']),
                'current_dots': None,
                'done': False,
            }

        # Fade in an "Explored: N%" label under each panel, seeded from the
        # starting coverage (before any exploration moves)
        init_texts = []
        for cfg in panel_configs:
            frac = self.coverage_percentage(
                cfg['state']['visited_positions'], cfg['fov_r'], cfg['area']
            )
            txt = self._make_pct_text(frac, cfg['label_anchor'])
            cfg['state']['pct_text'] = txt
            init_texts.append(FadeIn(txt))

            if cfg.get('mario_pos') is not None:
                found = self.is_point_covered(
                    cfg['mario_pos'], cfg['state']['visited_positions'], cfg['fov_r']
                )
                status_txt = self._make_status_text(found, txt)
                cfg['state']['status_text'] = status_txt
                init_texts.append(FadeIn(status_txt))
        if init_texts:
            self.play(*init_texts, run_time=0.4)

        for _ in range(max_moves):
            active = [cfg for cfg in panel_configs if not cfg['state']['done']]
            if not active:
                break

            steps = {}
            for cfg in active:
                res = self._frontier_step(cfg)
                steps[cfg['id']] = res
                if res['done']:
                    cfg['state']['done'] = True

            # Phase 1: swap in the recomputed frontier dots, together
            dot_anims = []
            for cfg in active:
                res = steps[cfg['id']]
                if res['done']:
                    continue
                old_dots = cfg['state']['current_dots']
                if old_dots is not None:
                    dot_anims.append(FadeOut(old_dots))
                dot_anims.append(FadeIn(res['new_dots_panel']))
                cfg['state']['current_dots'] = res['new_dots_panel']
            if dot_anims:
                self.play(*dot_anims, run_time=0.4)

            # Phase 2: move every drone of every panel + grow FoVs, all together
            move_anims = []
            for cfg in active:
                res = steps[cfg['id']]
                if res['done']:
                    continue
                panel = cfg['panel']
                for d_idx, tgt in res['drone_targets'].items():
                    move_anims.append(panel[1][d_idx].animate.move_to(tgt['target_pos_panel']))
                    move_anims.append(panel[2][d_idx].animate.move_to(tgt['target_pos_panel']))
                    cfg['state']['current_positions'][d_idx] = tgt['target_pos_world']
                    cfg['state']['visited_positions'].append(tgt['target_pos_world'])
                move_anims.append(Transform(panel[0], res['new_area_panel']))
                cfg['state']['known_world'] = res['new_known_world']

                new_frac = self.coverage_percentage(
                    cfg['state']['visited_positions'], cfg['fov_r'], cfg['area']
                )
                new_txt = self._make_pct_text(new_frac, cfg['label_anchor'])
                move_anims.append(Transform(cfg['state']['pct_text'], new_txt))

                if cfg.get('mario_pos') is not None:
                    found = self.is_point_covered(
                        cfg['mario_pos'], cfg['state']['visited_positions'], cfg['fov_r']
                    )
                    new_status_txt = self._make_status_text(found, new_txt)
                    move_anims.append(Transform(cfg['state']['status_text'], new_status_txt))
            if move_anims:
                self.play(*move_anims, run_time=0.8)

        # Final cleanup: fade out any panels' remaining frontier dots together
        cleanup = [
            FadeOut(cfg['state']['current_dots'])
            for cfg in panel_configs if cfg['state']['current_dots'] is not None
        ]
        if cleanup:
            self.play(*cleanup, run_time=0.5)

        return {cfg['id']: cfg['state']['known_world'] for cfg in panel_configs}

    def raw_info_gain(self, pts_xy, visited_positions, fov_r, area, sq_size=0.4, res=8):
        """Unnormalized information gain per point: fraction of a small square
        around it that's still unseen by ANY visited FoV position. Unlike
        get_info_gain_scales (which is stretched to a fixed 0.1-2.0 display
        range), this shrinks toward 0 as the area actually becomes known, so
        it can be used as a real stopping condition."""
        v = area.get_vertices()
        x0, x1 = v[:, 0].min(), v[:, 0].max()
        y0, y1 = v[:, 1].min(), v[:, 1].max()
        ctrs = np.array([p[:2] for p in visited_positions])

        gains = []
        for pt in pts_xy:
            xs = np.linspace(pt[0] - sq_size / 2, pt[0] + sq_size / 2, res)
            ys = np.linspace(pt[1] - sq_size / 2, pt[1] + sq_size / 2, res)
            xv, yv = np.meshgrid(xs, ys)
            grid = np.c_[xv.ravel(), yv.ravel()]
            in_area = (grid[:, 0] >= x0) & (grid[:, 0] <= x1) & \
                      (grid[:, 1] >= y0) & (grid[:, 1] <= y1)
            valid = grid[in_area]
            if len(valid) == 0:
                gains.append(0.0)
                continue
            dists = np.linalg.norm(valid[:, None, :] - ctrs[None, :, :], axis=2)
            in_fov = np.any(dists <= fov_r, axis=1)
            gains.append(np.sum(~in_fov) / len(grid))
        return np.array(gains)

    def make_drone(self, color):
        drone = SVGMobject("Drone.svg").scale(0.2).set_color(color)
        drone.z_index = 4
        return drone


    # ── panel builder ───────────────────────────────────────────────────────────

    def build_panel(self, drones, fov_rings, fov_area, filt_pts, dot_colors,
                    zoom_center, scale, target_cx, dot_radii=None, ellipses=None):
        """
        Clone drones, FoV area/rings, build dots, and optionally lay over ellipses.
        """
        zoom_cx, zoom_cy = zoom_center[0], zoom_center[1]

        if dot_radii is None:
            dot_radii = [0.055] * len(filt_pts)

        p_area   = fov_area.copy()
        p_drones = VGroup(*[d.copy() for d in drones])
        p_fovs   = VGroup(*[f.copy() for f in fov_rings])
        
        p_dots   = VGroup(*[
            Dot(pt, radius=r, color=col, z_index=3)
            for pt, col, r in zip(filt_pts, dot_colors, dot_radii)
        ])

        panel = VGroup(p_area, p_drones, p_fovs, p_dots)
        
        if ellipses is not None:
            panel.add(ellipses.copy())

        panel.scale(scale, about_point=zoom_center)
        panel.shift(np.array([target_cx - zoom_cx, -zoom_cy, 0]))
        return panel

    # ── construct ───────────────────────────────────────────────────────────────

    def construct(self):

        # ── 1. Rectangle ─────────────────────────────────────────────────────────
        area = Rectangle(width=7, height=4.8)
        area.set_stroke(WHITE, 2).set_fill("#1a1a2e", opacity=1)
        area.move_to(ORIGIN)
        area.z_index = 0
        text_label = Text("Unknown Area").next_to(area, UP)

        self.play(Create(area), Write(text_label), run_time=1.1)
        
        # Narrator: "Let us consider a scenario..."
        self.wait(1.0) 

        # ── 2. Mario at top-right ─────────────────────────────────────────────────
        # 1. Initialize Mario once
        mario_1 = ImageMobject("Luigi_Baloon.png").scale_to_fit_height(2)
        mario_label = Text("Luigi Status : ",color=GREEN).next_to(area, UP)
        mario = ImageMobject("Luigi.png").scale_to_fit_height(0.9)
        self.add(mario_1) # Use add for the initial state
        self.play(Transform(text_label,mario_label))
        self.wait(0.15)
        mario_label_new = Text("Luigi Status : Lost",color=RED).next_to(area, UP)
        self.play(
            FadeOut(mario_1) ,
            FadeIn(mario) ,
            mario.animate.move_to(area.get_corner(UR) + 0.6 * DL)
            .scale_to_fit_height(0.75)
            .set_z_index(6),
            Transform(text_label, mario_label_new),
            run_time=1.1
        )
        self.wait(2)

        # ── 3. Drones + FoV rings ─────────────────────────────────────────────────
        FOV_R   = 1.25
        COLORS  = [BLUE, RED, YELLOW_D]
        SPACING = 1.6
        bot_ctr = area.get_bottom() + 0.40 * UP
        offsets = [-SPACING, 0.0, SPACING]

        drones, fov_rings = [], []
        
        for dx, col in zip(offsets, COLORS):
            pos = bot_ctr + dx * RIGHT
            d   = self.make_drone(col)
            d.move_to(pos)
            f = Circle(radius=FOV_R, z_index=1)
            f.move_to(pos).set_stroke(col, width=1.4, opacity=0.55).set_fill(opacity=0)
            drones.append(d)
            fov_rings.append(f)
            
        for d, f in zip(drones, fov_rings):
            drone_FoV_Label = Text("Team of 3 Rescue Drones",color=d.color).next_to(area, UP)
            self.play(FadeIn(d),Transform(text_label,drone_FoV_Label), run_time=0.5)
        self.wait(0.3)

        for d, f in zip(drones, fov_rings):
            drone_FoV_Label = Text("Drone's Field Of View",color=d.color).next_to(area, UP)
            self.play(Create(f),Transform(text_label,drone_FoV_Label), run_time=0.5)
            
        # Narrator: "Let's use our 3 drones at our disposal, which has a circular fov."
        self.wait(1.0) 

        # ── 4. White FoV union ────────────────────────────────────────────────────
        fov_aoi_list = [self.drone_fov_aoi(f, area) for f in fov_rings]
        merged_label = Text("Merged Known Area",color=WHITE).next_to(area, UP)
        combined = Union(
            *fov_aoi_list,
            fill_color=WHITE, fill_opacity=0.25,
            stroke_width=0, z_index=2,
        )
        self.play(FadeIn(combined), Transform(text_label,merged_label), run_time=0.9)
        self.wait(0.3)

        # ── 5. Green perimeter → discretised dots ─────────────────────────────────
        frontier_line = combined.copy()
        frontier_line.set_fill(opacity=0).set_stroke(GREEN, width=2.5)
        frontier_line.z_index = 3
        continuous_frontier_label = Text("Frontier",color=GREEN).next_to(area, UP)
        self.play(Create(frontier_line),Transform(text_label,continuous_frontier_label), run_time=1.2)
        
        # Narrator: "A direct approach to exploring the area would be to analyze the frontiers, the boundary between the known and unknown."
        self.wait(2.0) 

        raw_pts  = self.sample_perimeter(combined, step=0.3)
        filt_pts = self.points_inside_area(raw_pts, area, margin=0.05)

        green_dots = VGroup(*[
            Dot(p, radius=0.055, color=GREEN, z_index=3) for p in filt_pts
        ])
        discrete_frontier_label = Text("Discretized Frontier",color=GREEN).next_to(area, UP)
        self.play(Transform(frontier_line, green_dots),Transform(text_label,discrete_frontier_label), run_time=1.0)
        
        # Narrator: "We can discretize the frontiers to generate points of interest."
        self.wait(2.0) 

        # # ── 6. Zoom into drone cluster ────────────────────────────────────────────
        drone_ctrs  = np.array([d.get_center() for d in drones])
        zoom_center = np.array([drone_ctrs[:, 0].mean(),
                                drone_ctrs[:, 1].mean(), 0])

        self.play(
            self.camera.frame.animate.move_to(zoom_center).set(width=6.5),
            run_time=1.5,
        )
        self.wait(0.6)
        self.remove(text_label)


        # ── 6.5 Arrows scanning from drones to dots (Post-Zoom) ───────────────────
        arrow_kwargs = dict(
            buff=0.22, color=LIGHT_GREY, stroke_width=1.5, 
            max_tip_length_to_length_ratio=0.05, z_index=3
        )
        
        scan_arrows = VGroup(*[
            Arrow(start=drones[0].get_center(), end=d.get_center(), **arrow_kwargs) 
            for d in green_dots
        ])
        self.play(Create(scan_arrows), run_time=1.0)
        self.wait(0.3)

        scan_arrows2 = VGroup(*[
            Arrow(start=drones[1].get_center(), end=d.get_center(), **arrow_kwargs) 
            for d in green_dots
        ])
        self.play(Transform(scan_arrows, scan_arrows2), run_time=0.8)
        self.wait(0.3)

        scan_arrows3 = VGroup(*[
            Arrow(start=drones[2].get_center(), end=d.get_center(), **arrow_kwargs) 
            for d in green_dots
        ])
        self.play(Transform(scan_arrows, scan_arrows3), run_time=0.8)
        self.wait(0.5)
        
        self.play(FadeOut(scan_arrows), run_time=0.5)
        self.wait(0.5)

        # # ── 7. Fade out scene ─────────────────────────────────────────────────────
        self.play(
            FadeOut(area), FadeOut(mario), FadeOut(combined),
            FadeOut(frontier_line),   
            *[FadeOut(d) for d in drones],
            *[FadeOut(f) for f in fov_rings],
            run_time=1.0,
        )
        self.wait(0.3)
        self.camera.frame.move_to(ORIGIN).set(width=config.frame_width)

        # ── 8. Compute clustering & Information Gain ──────────────────────────────
        pts_xy          = filt_pts[:, :2]
        drone_positions = [d.get_center() for d in drones]

        # Shared IG Scales
        ig_scales = self.get_info_gain_scales(
            pts_xy, drone_positions, FOV_R, area, sq_size=0.4, res=10
        )

        km_colors, _, _ = run_kmeans(pts_xy, drone_positions, COLORS, k=3)
        dp_colors, dp_means, dp_covars, dp_cluster_cols, dp_resp = run_dpgmm(pts_xy, drone_positions, COLORS, k=3)

        # Generate Gaussian overlay ellipses 
        gmm_ellipses = self.get_gmm_ellipses(dp_means, dp_covars, dp_cluster_cols, num_stdev=2.0)

        # Calculate radii scales based on the RED probability for the Right Panel
        try:
            red_idx = dp_cluster_cols.index(RED)
            red_probs = dp_resp[:, red_idx]
        except ValueError:
            red_probs = np.zeros(len(pts_xy))

        scale_factors = 0.1 + 1.9 * red_probs
        BASE_RADIUS = 0.055
        dp_radii = BASE_RADIUS * scale_factors

        # # ── 9. Build all four panel states ────────────────────────────────────────
        SCALE   = 1.19
        PANEL_X = 3.5

        # Initial: both panels with green dots and FoV area
        left_green  = self.build_panel(drones, fov_rings, combined, filt_pts,
                                       [GREEN] * len(filt_pts),
                                       zoom_center, SCALE, -PANEL_X)
        
        right_green = self.build_panel(drones, fov_rings, combined, filt_pts,
                                       [GREEN] * len(filt_pts),
                                       zoom_center, SCALE,  PANEL_X)

        # Target coloured panel for the Left
        left_km  = self.build_panel(drones, fov_rings, combined, filt_pts, km_colors,
                                    zoom_center, SCALE, -PANEL_X)
        
        # Target coloured panel for the Right
        right_dp = self.build_panel(drones, fov_rings, combined, filt_pts, dp_colors,
                                    zoom_center, SCALE,  PANEL_X,
                                    dot_radii=dp_radii)

        divider = DashedLine(
            UP * 3.8, DOWN * 3.8,
            dash_length=0.18, dashed_ratio=0.5,
            color=LIGHT_GREY, stroke_width=2, z_index=10,
        )

        lbl_l = (Text("Hard Clustering", font_size=28,
                       color=WHITE, weight=BOLD)
                 .scale(0.75).move_to(LEFT  * PANEL_X + UP * 3.3))
        lbl_r = (Text("Soft Clustering", font_size=28,
                       color=WHITE, weight=BOLD)
                 .scale(0.75).move_to(RIGHT * PANEL_X + UP * 3.3))

        # ── 10. Camera Sequenced Animation ────────────────────────────────────────

        # Step A: Show full split screen
        self.play(
            FadeIn(left_green), FadeIn(right_green),
            Create(divider),
            FadeIn(lbl_l), FadeIn(lbl_r),
            run_time=1.2,
        )
        self.wait(0.5)

        # Step B: Zoom into LEFT Panel (Hard Clustering)
        self.play(
            self.camera.frame.animate.move_to(LEFT * PANEL_X).set(width=8.0),
            run_time=1.5
        )
        self.wait(0.5)

        # Step C: LEFT Sequence — hard-coloured dots
        self.play(Transform(left_green, left_km), run_time=1.1)
        
        # Narrator: "Currently exploration approaches take a direct hard clustering approach to delegate the frontiers, this gives hard partitions that doesn't confuse the drones."
        self.wait(2.0) 

        # Step C.0: LEFT Sequence — showcase each cluster in turn:
        cluster_name      = {BLUE: "Blue", RED: "Red", YELLOW_D: "Yellow"}
        cluster_order     = [RED, BLUE, YELLOW_D]
        BASE_LEFT_RADIUS  = 0.055 * SCALE

        for c in cluster_order:
            drone_idx    = COLORS.index(c)
            this_drone   = left_green[1][drone_idx]
            other_drones = [left_green[1][i] for i in range(len(COLORS)) if i != drone_idx]

            member_mask = np.array([col == c for col in km_colors])
            pop_factors = np.where(member_mask, 1.8, 0.55) 

            cluster_label = Text(
                f"{cluster_name[c]} cluster — hard assignment",
                font_size=24, color=c, weight=BOLD,
            ).next_to(LEFT * PANEL_X, UP, buff=2.6)

            self.play(
                *[m.animate.set_opacity(0.25) for m in other_drones],
                Write(cluster_label),
                run_time=0.7,
            )

            self.play(
                Indicate(this_drone, scale_factor=1.5, color=c),
                *[dot.animate.scale(f) for dot, f in zip(left_green[3], pop_factors)],
                run_time=1.2,
            )
            self.wait(0.8)

            self.play(
                *[m.animate.set_opacity(1) for m in other_drones],
                *[dot.animate.scale(1 / f) for dot, f in zip(left_green[3], pop_factors)],
                FadeOut(cluster_label),
                run_time=0.6,
            )

        # Step C.1: Find the closest red/yellow frontier pair
        red_mask    = np.array([c == RED      for c in km_colors])
        yellow_mask = np.array([c == YELLOW_D for c in km_colors])
        red_pts_xy    = pts_xy[red_mask]
        yellow_pts_xy = pts_xy[yellow_mask]

        def to_left_panel(pt_xy):
            world  = np.array([pt_xy[0], pt_xy[1], 0.0])
            scaled = zoom_center + (world - zoom_center) * SCALE
            return scaled + np.array([-PANEL_X - zoom_center[0], -zoom_center[1], 0.0])

        if len(red_pts_xy) and len(yellow_pts_xy):
            ry_dists = np.linalg.norm(
                red_pts_xy[:, None, :] - yellow_pts_xy[None, :, :], axis=2
            )
            r_idx, y_idx = np.unravel_index(np.argmin(ry_dists), ry_dists.shape)
            red_boundary_pt    = red_pts_xy[r_idx]
            yellow_boundary_pt = yellow_pts_xy[y_idx]

            red_boundary_panel    = to_left_panel(red_boundary_pt)
            yellow_boundary_panel = to_left_panel(yellow_boundary_pt)
            boundary_mid_panel    = (red_boundary_panel + yellow_boundary_panel) / 2

            # width=3.2 previously gave a frame height of only ~1.8 (16:9),
            # so the boundary/overlap labels (placed buff=0.9 above/below
            # center) fell partly or fully outside the visible camera —
            # widen the zoom so both labels stay comfortably in frame.
            self.play(
                self.camera.frame.animate.move_to(boundary_mid_panel).set(width=6.4),
                run_time=1.3,
            )
            self.wait(0.4)

            red_drone_panel,    red_fov_panel    = left_green[1][1], left_green[2][1]
            yellow_drone_panel, yellow_fov_panel = left_green[1][2], left_green[2][2]
            red_home_pos    = red_drone_panel.get_center()
            yellow_home_pos = yellow_drone_panel.get_center()

            boundary_label = Text(
                "Defined boundary",
                font_size=24, color=WHITE,
            ).next_to(boundary_mid_panel, UP, buff=0.9)
            self.play(Write(boundary_label), run_time=0.8)

            self.play(
                red_drone_panel.animate.move_to(red_boundary_panel),
                red_fov_panel.animate.move_to(red_boundary_panel),
                yellow_drone_panel.animate.move_to(yellow_boundary_panel),
                yellow_fov_panel.animate.move_to(yellow_boundary_panel),
                run_time=1.3,
            )
            self.wait(0.3)

            overlap_region = Intersection(
                red_fov_panel, yellow_fov_panel,
                fill_color=ORANGE, fill_opacity=0.55, stroke_width=0, z_index=5,
            )
            overlap_label = Text(
                "Overlapping coverage",
                font_size=24, color=ORANGE, weight=BOLD,
            ).next_to(boundary_mid_panel, DOWN, buff=0.9)

            self.play(
                Transform(boundary_label, overlap_label),
                FadeIn(overlap_region),
                run_time=1.0,
            )
            
            # Narrator: "However, this means that they can overlap and chase nearby frontiers, causing overlap and possible collision."
            self.wait(3.5) 

            self.play(
                FadeOut(overlap_region), FadeOut(boundary_label),
                red_drone_panel.animate.move_to(red_home_pos),
                red_fov_panel.animate.move_to(red_home_pos),
                yellow_drone_panel.animate.move_to(yellow_home_pos),
                yellow_fov_panel.animate.move_to(yellow_home_pos),
                run_time=1.0,
            )
            self.play(
                self.camera.frame.animate.move_to(LEFT * PANEL_X).set(width=8.0),
                run_time=1.3,
            )
            self.wait(0.4)

        # Step E: Zoom out to center 
        self.play(
            self.camera.frame.animate.move_to(ORIGIN).set(width=config.frame_width),
            run_time=1.5
        )
        self.wait(0.5)

        # Step F: Zoom into RIGHT Panel (Soft Clustering)
        self.play(
            self.camera.frame.animate.move_to(RIGHT * PANEL_X).set(width=8.0),
            run_time=1.5
        )
        self.wait(0.5)

        # Step G: RIGHT Sequence — soft-blended dots (with probability scaling)
        self.play(Transform(right_green, right_dp), run_time=1.3)
        self.wait(0.5)

        # Step H: RIGHT Sequence — Fade in ellipses
        gmm_ellipses.scale(SCALE, about_point=zoom_center)
        gmm_ellipses.shift(np.array([PANEL_X - zoom_center[0], -zoom_center[1], 0]))
        self.play(FadeIn(gmm_ellipses), run_time=1.0)
        
        # Narrator: "To avoid this collision, we use soft clustering."
        self.wait(2.0)

        # Step H.1: RIGHT Sequence — showcase each cluster in turn
        def dim_ellipse(e):
            return e.animate.set_fill(opacity=0.05).set_stroke(opacity=0.3)

        def restore_ellipse(e):
            return e.animate.set_fill(opacity=0.2).set_stroke(opacity=1)

        cluster_name = {BLUE: "Blue", RED: "Red", YELLOW_D: "Yellow"}
        cluster_order = [RED, BLUE, YELLOW_D]

        current_radii = dp_radii * SCALE

        for c in cluster_order:
            comp_idx  = dp_cluster_cols.index(c)
            drone_idx = COLORS.index(c)

            this_drone     = right_green[1][drone_idx]
            this_ellipse   = gmm_ellipses[comp_idx]
            other_drones   = [right_green[1][i] for i in range(len(COLORS)) if i != drone_idx]
            other_ellipses = [gmm_ellipses[i] for i in range(len(gmm_ellipses)) if i != comp_idx]

            probs           = dp_resp[:, comp_idx]
            scale_factors_c = 0.1 + 1.9 * probs
            target_radii_c  = BASE_RADIUS * scale_factors_c * SCALE

            cluster_label = Text(
                f"{cluster_name[c]} cluster — soft membership",
                font_size=24, color=c, weight=BOLD,
            ).next_to(RIGHT * PANEL_X, UP, buff=2.6)

            self.play(
                *[m.animate.set_opacity(0.25) for m in other_drones],
                *[dim_ellipse(e) for e in other_ellipses],
                Write(cluster_label),
                run_time=0.7,
            )

            dot_scale_anims = [
                dot.animate.scale(target_r / current_r)
                for dot, target_r, current_r in zip(right_green[3], target_radii_c, current_radii)
            ]
            self.play(
                Indicate(this_drone, scale_factor=1.5, color=c),
                this_ellipse.animate.set_stroke(width=4),
                *dot_scale_anims,
                run_time=1.2,
            )
            current_radii = target_radii_c
            self.wait(0.8)

            self.play(
                this_ellipse.animate.set_stroke(width=2.5),
                *[m.animate.set_opacity(1) for m in other_drones],
                *[restore_ellipse(e) for e in other_ellipses],
                FadeOut(cluster_label),
                run_time=0.6,
            )

        dot_restore_anims = [
            dot.animate.scale(target_r / current_r)
            for dot, target_r, current_r in zip(right_green[3], dp_radii * SCALE, current_radii)
        ]
        self.play(*dot_restore_anims, run_time=1.0)
        
        # Narrator: "Where there is no hard clustering involved, rather each frontier belongs to all drones in a probabilistic sense."
        self.wait(3.0) 

        # Step I: Zoom out to see both panels, then run Information Gain
        # simultaneously on both — same ig_scales apply identically to hard
        # and soft clustering, so it's shown once, side by side, rather than
        # twice while zoomed into each panel separately.
        # Step J: Fade out ovals before zooming out
        self.play(
            FadeOut(gmm_ellipses),
            self.camera.frame.animate.move_to(ORIGIN).set(width=config.frame_width),
            run_time=1.5
        )
        self.wait(0.3)

        def make_ig_squares(panel_x):
            squares = VGroup(*[
                Square(side_length=0.4, stroke_color=YELLOW, stroke_width=2)
                .move_to(np.array([pt[0], pt[1], 0]))
                for pt in pts_xy
            ])
            squares.scale(SCALE, about_point=zoom_center)
            squares.shift(np.array([panel_x - zoom_center[0], -zoom_center[1], 0]))
            return squares

        left_ig_squares  = make_ig_squares(-PANEL_X)
        right_ig_squares = make_ig_squares(PANEL_X)

        self.play(
            Create(left_ig_squares),
            Create(right_ig_squares),
            run_time=1.0
        )

        left_scale_anims = [
            dot.animate.scale(ig) for dot, ig in zip(left_green[3], ig_scales)
        ]
        right_scale_anims = [
            dot.animate.scale(ig) for dot, ig in zip(right_green[3], ig_scales)
        ]
        self.play(
            FadeOut(left_ig_squares),
            FadeOut(right_ig_squares),
            *left_scale_anims,
            *right_scale_anims,
            run_time=1.2
        )

        # Narrator: "The prioritization is done by information gain — the amount of knowledge gained by visiting a point — and it applies identically whether drones are hard- or soft-clustered, with soft clustering minimizing the overlaps."
        self.wait(3.0) 

        

        # Step K: Zoom out to show initial rectangle and mario on both sides, fade out dots
        
        LEFT_SHIFT  = np.array([-1.5,  -1.5, 0])
        RIGHT_SHIFT = np.array([ 1.5, -1.5, 0])
        MAX_DRONE_STEP = 1.0
        EDGE_BOOST = 1.0

        area_left = area.copy().scale(SCALE, about_point=zoom_center).shift(np.array([-PANEL_X - zoom_center[0], -zoom_center[1], 0]))
        mario_left = mario.copy().scale(SCALE, about_point=zoom_center).shift(np.array([-PANEL_X - zoom_center[0], -zoom_center[1], 0]))
        area_right = area.copy().scale(SCALE, about_point=zoom_center).shift(np.array([PANEL_X - zoom_center[0], -zoom_center[1], 0]))
        mario_right = mario.copy().scale(SCALE, about_point=zoom_center).shift(np.array([PANEL_X - zoom_center[0], -zoom_center[1], 0]))
        
        area_left.z_index = 0
        area_right.z_index = 0
        mario_left.z_index = 6
        mario_right.z_index = 6

        area_left.shift(LEFT_SHIFT)
        mario_left.shift(LEFT_SHIFT)
        area_right.shift(RIGHT_SHIFT)
        mario_right.shift(RIGHT_SHIFT)

        self.play(
            self.camera.frame.animate.move_to(ORIGIN).set(width=20.0),
            FadeIn(area_left), FadeIn(mario_left),
            FadeIn(area_right), FadeIn(mario_right),
            left_green.animate.shift(LEFT_SHIFT),
            right_green.animate.shift(RIGHT_SHIFT),
            run_time=2.0
        )
        self.wait(0.5)

        # ── Step L: Drone exploration (synchronized across both panels) ─────────
        self.play(FadeOut(left_green[3]), FadeOut(right_green[3]), run_time=0.6)

        mario_world_pos = mario.get_center()

        self.explore_synced([
            dict(
                id='left',
                panel=left_green,
                area=area,
                fov_r=FOV_R,
                zoom_center=zoom_center,
                scale=SCALE,
                target_cx=-PANEL_X,
                extra_shift=LEFT_SHIFT,
                mode="hard",
                colors=COLORS,
                drone_positions_start=drone_positions,
                known_world=combined.copy(),
                label_anchor=area_left,
                edge_boost=EDGE_BOOST,
                max_step=MAX_DRONE_STEP,
                mario_pos=mario_world_pos,
            ),
            dict(
                id='right',
                panel=right_green,
                area=area,
                fov_r=FOV_R,
                zoom_center=zoom_center,
                scale=SCALE,
                target_cx=PANEL_X,
                extra_shift=RIGHT_SHIFT,
                mode="soft",
                colors=COLORS,
                drone_positions_start=drone_positions,
                known_world=combined.copy(),
                label_anchor=area_right,
                max_step=MAX_DRONE_STEP,
                mario_pos=mario_world_pos,
            ),
        ])
        
        # Narrator: "Furthermore, our approach is method agnostic, and thus can be plugged into any existing frontier exploration methods to improve its performance."
        self.wait(5.0) 

        # ── Closing Sequence ─────────────────────────────────────────────
        # Step M: Clean the screen entirely before the closing slide
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=1.0)
        self.camera.frame.move_to(ORIGIN).set(width=config.frame_width)
        self.wait(0.3)

        # Step N: "Real World Concerns" title + bullet points
        concerns_title = Text("Real World Concerns", font_size=40, weight=BOLD, color=WHITE)
        concerns_title.to_edge(UP, buff=1.0)
        concerns_title.to_edge(LEFT, buff=1.0)

        concern_items = [
            "Sensor Errors",
            "Communication Delays",
            "Obstacles",
            "Incorrect Coordinations",
        ]

        def make_star_bullet_row(text_str):
            star = ImageMobject("mario-star.png").scale_to_fit_height(0.34)
            label = Text(text_str, font_size=36, color=WHITE)
            label.next_to(star, RIGHT, buff=0.25)
            # Group (not VGroup) since this mixes an ImageMobject with a Text/VMobject
            return Group(star, label)

        concerns_list = Group(*[make_star_bullet_row(t) for t in concern_items])
        concerns_list.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        concerns_list.next_to(concerns_title, DOWN, buff=0.8)
        concerns_list.to_edge(LEFT, buff=1.0)

        self.play(Write(concerns_title), run_time=1.0)
        self.play(
            LaggedStart(
                *[FadeIn(item, shift=RIGHT * 0.3) for item in concerns_list],
                lag_ratio=0.3,
            ),
            run_time=2.0,
        )
        
        # Narrator: "A real concern would be issues of sensory and mapping errors, communication delays, obstacles and coordination miscalculations."
        self.wait(4.0) 

        # Step O: Clean the screen again before the closing slide
        self.play(FadeOut(concerns_title), FadeOut(concerns_list), run_time=1.0)
        self.wait(0.3)
        # Step P: "Thank you" + QR codes for the paper and video
        thank_you = Text("Thank you", font_size=54, weight=BOLD, color=WHITE)
        thank_you.to_edge(UP, buff=1.2)
        paper_qr   = ImageMobject("Paper_Link_invert.png").scale_to_fit_height(2.6)
        youtube_qr = ImageMobject("Youtube_Link_invert.png").scale_to_fit_height(2.6)
        castle = ImageMobject("mario_Castle.png").scale_to_fit_height(2.6)
        paper_qr.move_to(LEFT * 3)
        youtube_qr.move_to(RIGHT * 3)

        # Labels are "popped" out by Luigi jump-hitting each QR code, so their
        # resting spot is just above it (mirrors a Mario block-hit coin pop).
        paper_label   = Text("Paper", font_size=28, color=WHITE).next_to(paper_qr, UP, buff=-0.2)
        youtube_label = Text("Video", font_size=28, color=WHITE).next_to(youtube_qr, UP, buff=-0.2)
        paper_label_target   = paper_label.get_center()
        youtube_label_target = youtube_label.get_center()

        # Captions that appear underneath each QR code once Luigi has hit it
        paper_caption = Text("Mathematical Stuff", font_size=22, color=WHITE).next_to(paper_qr, DOWN, buff=0.25)
        youtube_caption = Text("Dual Drone \nReal-World \nExperiments", font_size=22, color=WHITE).next_to(youtube_qr, DOWN, buff=0.25)
        paper_caption.set_opacity(0)
        youtube_caption.set_opacity(0)
        self.add(paper_caption, youtube_caption)

        self.play(Write(thank_you), run_time=1.0)
        self.play(
            FadeIn(paper_qr, shift=UP * 0.3),
            FadeIn(youtube_qr, shift=UP * 0.3),
            FadeIn(castle),
            run_time=1.2,
        )
        self.wait(0.3)

        # Luigi runs in from off-screen left along the ground beneath the QR codes
        GROUND_Y   = paper_qr.get_bottom()[1] - 1.4
        JUMP_SHIFT = paper_qr.get_bottom()[1] - GROUND_Y - 0.15  # stop just shy of the QR's underside
        CASTLE_CENTER = castle.get_center()

        luigi = ImageMobject("Luigi.png").scale_to_fit_height(1.1)
        luigi.move_to(np.array([-config.frame_width / 2 - 1.0, GROUND_Y, 0]))
        luigi.z_index = 10
        self.add(luigi)

        # Hide the labels inside their QR codes until Luigi "hits" them out
        paper_label.move_to(paper_qr.get_top()).set_opacity(0)
        youtube_label.move_to(youtube_qr.get_top()).set_opacity(0)
        self.add(paper_label, youtube_label)

        def run_and_jump(target_x, qr, label, label_target, caption):
            # Run along the ground to just below the QR code
            self.play(
                luigi.animate.move_to(np.array([target_x, GROUND_Y, 0])),
                run_time=1.0,
                rate_func=linear,
            )
            # Jump straight up to bump the QR code from below
            self.play(
                luigi.animate.shift(UP * JUMP_SHIFT),
                run_time=0.25,
                rate_func=rush_from,
            )
            # Land back down while the QR shakes from the impact and the label
            # pops out the top of it, overshooting slightly
            self.play(
                luigi.animate.shift(DOWN * JUMP_SHIFT),
                label.animate.move_to(label_target + UP * 0.15).set_opacity(1),
                Wiggle(qr, scale_value=1.04, rotation_angle=0.02 * TAU, n_wiggles=6, run_time=0.25),
                run_time=0.25,
                rate_func=rush_into,
            )
            # Settle the label into its final resting spot — the little bounce-back
            # and fade in the caption underneath as Luigi's about to move on
            self.play(
                label.animate.move_to(label_target),
                caption.animate.set_opacity(0),
                run_time=0.15,
            )

        run_and_jump(paper_qr.get_x(),   paper_qr,   paper_label,   paper_label_target,   paper_caption)
        run_and_jump(youtube_qr.get_x(), youtube_qr, youtube_label, youtube_label_target, youtube_caption)

        # Luigi jogs back toward the castle and disappears inside it
        self.play(
            luigi.animate.move_to(np.array([CASTLE_CENTER[0], GROUND_Y, 0])),
            run_time=1.2,
            rate_func=linear,
        )
        self.play(
            luigi.animate.move_to(CASTLE_CENTER).scale(0.3),
            FadeOut(luigi),
            run_time=0.6,
            rate_func=smooth,
        )

        # Narrator: "To understand how we tackle these issues, please visit my poster. Thanks!"
        self.wait(5.0)
