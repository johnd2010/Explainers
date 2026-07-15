from manim import *
from manim_slides import Slide
import numpy as np
from sklearn.cluster import KMeans


class BaseSlide(Slide, MovingCameraScene):
    def setup(self):
        super().setup()
        self.add_logos()

    def add_logos(self):
        self.ulisboa = ImageMobject("./DPGMM/Logos/ULISBOA_VERTICAL_NEG_RGB.png")
        self.ist     = ImageMobject("./DPGMM/Logos/IST_A_RGB_NEG.png")
        self.larsys  = ImageMobject("./DPGMM/Logos/LARSYS_logo_black.png")
        self.isr     = ImageMobject("./DPGMM/Logos/ISR.png")
        self.it      = ImageMobject("./DPGMM/Logos/IT_Cleaned.png")

        self.header = Rectangle(
            width=config.frame_width, height=0.8,
            fill_color="#009de0", fill_opacity=1, stroke_color=None, z_index=-200
        ).to_edge(UP, buff=0)

        self.footer = Rectangle(
            width=config.frame_width, height=0.8,
            fill_color="#009de0", fill_opacity=1, stroke_color=None, z_index=-200
        ).to_edge(DOWN, buff=0)

        self.ulisboa.scale_to_fit_height(1.1 * self.header.height)
        self.ist.scale_to_fit_height(1.8 * self.header.height)
        self.ist.move_to(self.header.get_left() - 0.05 * self.header.width * LEFT)
        self.ulisboa.move_to(self.header.get_left() - 0.12 * self.header.width * LEFT)

        self.larsys.scale_to_fit_height(1.35 * self.header.height)
        self.larsys.move_to(self.header.get_right() - 0.1 * self.header.width * RIGHT)

        self.it.scale_to_fit_height(1.4 * self.footer.height)
        self.it.move_to(self.footer.get_left() - 0.07 * self.footer.width * LEFT)

        self.isr.scale_to_fit_height(0.8 * self.footer.height)
        self.isr.move_to(self.footer.get_right() - 0.1 * self.footer.width * RIGHT)

        self.add(self.header, self.footer, self.ulisboa, self.larsys, self.it, self.isr, self.ist)


class Presentation(BaseSlide):

    # ──────────────────────────────────────────────────
    #  Perimeter / geometry helpers
    # ──────────────────────────────────────────────────

    def sample_perimeter(self, vmobj: VMobject, step: float = 0.25) -> np.ndarray:
        points = vmobj.get_points()
        if len(points) < 2:
            return points.copy()
        diffs    = np.diff(points, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        cumlen   = np.concatenate([[0], np.cumsum(seg_lens)])
        total    = cumlen[-1]
        positions = np.arange(0, total, step)
        positions[-1] = min(positions[-1], total)
        out = []
        for t in positions:
            idx = min(np.searchsorted(cumlen, t, side="right") - 1, len(seg_lens) - 1)
            sl  = seg_lens[idx]
            a   = (t - cumlen[idx]) / sl if sl > 0 else 0.0
            out.append(points[idx] + a * (points[idx + 1] - points[idx]))
        return np.array(out)

    def set_submobject_fill_opacity(self, mobject, opacity):
        if hasattr(mobject, "submobjects"):
            for sub in mobject.submobjects:
                if hasattr(sub, "fill_opacity"):
                    sub.set_fill(opacity=opacity)
                self.set_submobject_fill_opacity(sub, opacity)
        elif hasattr(mobject, "fill_opacity"):
            mobject.set_fill(opacity=opacity)

    def drone_FoV(self, FoV, Area):
        return Intersection(
            FoV, Area,
            fill_color=WHITE, fill_opacity=0.5, stroke_width=0, z_index=Area.z_index
        )

    def drone_generator(self, Area, DRONE_COLOR=RED):
        drone = SVGMobject("DPGMM/Drone.svg").scale(0.1).set_color(DRONE_COLOR)
        drone.z_index = 2
        drone.move_to(Area.get_edge_center(DOWN + LEFT)).shift(0.3 * UP + 0.3 * RIGHT)
        FoV = Circle(radius=0.5, z_index=1).move_to(drone.get_center())
        return drone, FoV

    def points_bounded_by_area(self, points, area):
        corners = area.get_vertices()
        x_min, x_max = np.min(corners[:, 0]), np.max(corners[:, 0])
        y_min, y_max = np.min(corners[:, 1]), np.max(corners[:, 1])
        mask = (
            (points[:, 0] >= x_min + 0.1) & (points[:, 0] <= x_max - 0.1) &
            (points[:, 1] >= y_min + 0.1) & (points[:, 1] <= y_max - 0.1)
        )
        return points[mask]

    # ──────────────────────────────────────────────────
    #  Clustering helpers
    # ──────────────────────────────────────────────────

    def _kmeans_data(self, points, drone_positions, drone_colors, dot_radius=0.02):
        """
        Run KMeans(k=3). Each cluster's color = the drone whose centre is
        closest to that cluster centroid.
        Returns: dots VGroup, stars VGroup, cluster_colors list, labels array, centers (k,2)
        """
        k      = 4
        X      = points[:, :2]
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        centers = km.cluster_centers_               # (3, 2)

        drone_xy = np.array([d[:2] for d in drone_positions])
        cluster_colors = [
            drone_colors[int(np.argmin(np.linalg.norm(drone_xy - c, axis=1)))]
            for c in centers
        ]

        dots = VGroup(*[
            Dot(pt, radius=dot_radius, color=cluster_colors[lbl])
            for pt, lbl in zip(points, labels)
        ])
        
        return dots, cluster_colors, labels, centers

    def _dpgmm_dots(self, points, centers, cluster_colors, dot_radius=0.02):
        """
        Soft-assignment coloring via inverse-distance blending of cluster colors.
        Boundary points blend smoothly between their two nearest cluster colors.
        """
        dots = VGroup()
        for pt in points:
            xy    = pt[:2]
            dists = np.linalg.norm(centers - xy, axis=1)
            dists = np.where(dists < 1e-6, 1e-6, dists)
            w     = 1.0 / dists;  w /= w.sum()
            r = g = b = 0.0
            for wi, col in zip(w, cluster_colors):
                cr, cg, cb = color_to_rgb(col)
                r += wi * cr;  g += wi * cg;  b += wi * cb
            dots.add(Dot(pt, radius=dot_radius, color=rgb_to_color([r, g, b])))
        return dots

    # ──────────────────────────────────────────────────
    #  Split-screen clustering
    # ──────────────────────────────────────────────────

    def _show_split_clustering(
        self, *,
        drone_1, drone_2, drone_3,
        drone_colors,
        points,               # frontier points in original scene coords
        frontier_dots,        # VGroup of green dots already on screen
        known_area,           # grey blob — copied to each panel
        area_original,        # the original Rectangle — faded out, recreated ×2 on zoom-out
        person,               # person image — faded out
        zoomed_cam_center,    # np.ndarray: where camera is while zoomed in
        split_scale: float = 1.0,  # uniform scale applied to all panel objects
    ):
        # ── 1. Panel spacing derived from drone bounding box ──
        drone_centers  = np.array([d.get_center() for d in [drone_1, drone_2, drone_3]])
        cluster_span   = np.max(drone_centers[:, 0]) - np.min(drone_centers[:, 0])
        panel_gap      = cluster_span * 0.6 + 0.3
        drone_group_cx = np.mean(drone_centers[:, 0])

        target_x_right = zoomed_cam_center[0] + panel_gap / 2 + cluster_span / 2
        target_x_left  = zoomed_cam_center[0] - panel_gap / 2 - cluster_span / 2

        shift_r = np.array([target_x_right - drone_group_cx, 0, 0])
        shift_l = np.array([target_x_left  - drone_group_cx, 0, 0])

        # ── 2. Build clustering data ──
        drone_positions = [d.get_center() for d in [drone_1, drone_2, drone_3]]
        km_dots,  cluster_colors, km_labels, km_centers = \
            self._kmeans_data(points, drone_positions, drone_colors)
        dpgmm_dots_orig = self._dpgmm_dots(points, km_centers, cluster_colors)

        # Helper: scale a copy around its own centre, then shift to panel
        def _scaled_copy(mob, shift):
            return mob.copy().scale(split_scale).shift(shift)

        # Final colored dot groups — also need scaling (points are in scene coords)
        # Scale the points themselves so dots land in the right place after scaling
        group_center = np.array([drone_group_cx, np.mean(drone_centers[:, 1]), 0])

        def _scale_points(pts):
            """Scale point positions around the drone group centre."""
            return group_center + (pts - group_center) * split_scale

        scaled_pts_r = _scale_points(points) + shift_r
        scaled_pts_l = _scale_points(points) + shift_l

        km_dots_r  = VGroup(*[
            Dot(p, radius=0.02 * split_scale, color=km_dots[i].get_color())
            for i, p in enumerate(scaled_pts_r)
        ])
        # km_stars_r = km_stars.copy().scale(split_scale).shift(shift_r)

        dpgmm_dots_l = VGroup(*[
            Dot(p, radius=0.02 * split_scale, color=dpgmm_dots_orig[i].get_color())
            for i, p in enumerate(scaled_pts_l)
        ])

        # ── 3. Copies of scene objects for each panel (scaled then shifted) ──
        d1r = _scaled_copy(drone_1, shift_r);  d1l = _scaled_copy(drone_1, shift_l)
        d2r = _scaled_copy(drone_2, shift_r);  d2l = _scaled_copy(drone_2, shift_l)
        d3r = _scaled_copy(drone_3, shift_r);  d3l = _scaled_copy(drone_3, shift_l)

        known_r = _scaled_copy(known_area, shift_r)
        known_l = _scaled_copy(known_area, shift_l)

        # Green frontier dots scaled to match
        green_dots_r = VGroup(*[
            Dot(p, radius=0.02 * split_scale, color=GREEN)
            for p in scaled_pts_r
        ])
        green_dots_l = VGroup(*[
            Dot(p, radius=0.02 * split_scale, color=GREEN)
            for p in scaled_pts_l
        ])

        # ── 4. Divider ──
        div_x = zoomed_cam_center[0]
        div_y = zoomed_cam_center[1]
        div_h = self.camera.frame.height * 0.9
        divider = DashedLine(
            start=[div_x, div_y + div_h / 2, 0],
            end  =[div_x, div_y - div_h / 2, 0],
            dash_length=0.04, dashed_ratio=0.5,
            color=LIGHT_GREY, stroke_width=2, z_index=10,
        )

        # ── 5. Labels ──
        lbl_y  = div_y + div_h * 0.5
        lbl_sc = 0.08
        lbl_km = (Text("Conventional : Hard Clustering", font_size=28, color=WHITE, weight=BOLD)
                  .scale(lbl_sc).move_to([target_x_right, lbl_y, 0]))
        lbl_dp = (Text("Proposed : Soft Clustering",   font_size=28, color=WHITE, weight=BOLD)
                  .scale(lbl_sc).move_to([target_x_left,  lbl_y, 0]))

        # ── 6. Camera width for split view ──
        split_cam_width = abs(target_x_right - target_x_left) + cluster_span + 0.5
        split_cam_cx    = (target_x_right + target_x_left) / 2

        # ── Step A: Fade out single-area objects ──
        self.play(
            FadeOut(area_original),
            FadeOut(person),
            FadeOut(drone_1), FadeOut(drone_2), FadeOut(drone_3),
            # Fade the original known_area and frontier dots simultaneously
            FadeOut(known_area),
            FadeOut(frontier_dots),
            run_time=0.6,
        )

        # ── Step B: Both panels appear with drones + grey blob + green dots ──
        self.play(
            # Right panel
            FadeIn(known_r),
            FadeIn(d1r), FadeIn(d2r), FadeIn(d3r),
            FadeIn(green_dots_r),
            # Left panel
            FadeIn(known_l),
            FadeIn(d1l), FadeIn(d2l), FadeIn(d3l),
            FadeIn(green_dots_l),
            # Divider + labels
            Create(divider),
            FadeIn(lbl_km), FadeIn(lbl_dp),
            # Widen camera to show both panels
            self.camera.frame.animate
                .move_to([split_cam_cx, div_y, 0])
                .set(width=split_cam_width),
            run_time=1.2,
        )
        self.next_slide()

        # ── Step C: Right — green dots Transform into KMeans colored dots ──
        self.play(Transform(green_dots_r, km_dots_r), run_time=0.9)
        # self.play(
        #     LaggedStart(*[GrowFromCenter(s) for s in km_stars_r], lag_ratio=0.35),
        #     run_time=0.6,
        # )
        self.next_slide()

        # ── Step D: Left — green dots Transform into DPGMM gradient dots ──
        self.play(Transform(green_dots_l, dpgmm_dots_l), run_time=1.2)
        self.next_slide()

        # ── Step E: Zoom out — reveal the two area rectangles ──
        # area_w = area_original.width
        # area_r = area_original.copy().shift(shift_r)
        # area_l = area_original.copy().shift(shift_l)
        # self.add(area_r, area_l)   # silently added while off-screen

        # zoom_out_width = abs(target_x_right - target_x_left) + area_w + 1.5
        # self.play(
        #     self.camera.frame.animate
        #         .move_to([split_cam_cx, div_y, 0])
        #         .set(width=zoom_out_width),
        #     run_time=1.5,
        # )
        # self.next_slide()

    # ──────────────────────────────────────────────────
    #  construct / show_introduction
    # ──────────────────────────────────────────────────

    def construct(self):
        self.show_introduction()

    def show_introduction(self):
        title = Text("Robotics in Exploration and Rescue Operations", font_size=40).move_to(UP)
        self.play(Write(title))
        self.wait(0.5)
        self.next_slide()

        self.play(title.animate.shift(UP * 1.8).shift(LEFT * 1).set_opacity(0.9))
        self.wait(0.5)
        self.next_slide()

        Area = Rectangle(width=6.5, height=4.5).move_to(ORIGIN).shift(2 * RIGHT)
        self.play(Create(Area))
        self.next_slide()

        drone_1, drone_1_FoV = self.drone_generator(Area, BLUE)
        drone_2, drone_2_FoV = self.drone_generator(Area, RED)
        drone_3, drone_3_FoV = self.drone_generator(Area, YELLOW)
        drone_2.shift(0.5 * RIGHT);  drone_2_FoV.shift(0.5 * RIGHT)
        drone_3.shift(RIGHT);        drone_3_FoV.shift(RIGHT)
        drone_colors = [BLUE, RED, YELLOW]

        drone_1_FoV_AoI = self.drone_FoV(drone_1_FoV, Area)
        drone_2_FoV_AoI = self.drone_FoV(drone_2_FoV, Area)
        drone_3_FoV_AoI = self.drone_FoV(drone_3_FoV, Area)

        person = ImageMobject("DPGMM/Mario.png").scale(0.05)
        person.move_to(Area.get_edge_center(UP + RIGHT)).shift(0.3 * DOWN + 0.3 * LEFT)
        self.play(FadeIn(person))
        self.next_slide()

        self.play(
            FadeIn(drone_1.scale(0.5)),
            FadeIn(drone_2.scale(0.5)),
            FadeIn(drone_3.scale(0.5)),
            self.camera.frame.animate.move_to(drone_3.get_center()).scale(0.2),
            run_time=1.5,
        )
        self.next_slide()

        known_area = Union(drone_1_FoV_AoI, drone_2_FoV_AoI, drone_3_FoV_AoI, fill_opacity=0.3)
        known_area.set_fill(opacity=0.3)
        known_area.set_stroke(opacity=0.3)
        # known_area.set_stroke(opacity=0.3)
        self.play(FadeIn(known_area))

        frontier_line = known_area.copy().set_fill(opacity=0).set_stroke(color=GREEN, width=0.1)
        self.play(Create(frontier_line))
        self.next_slide()

        points    = self.sample_perimeter(known_area, step=0.1)
        points    = self.points_bounded_by_area(points, Area)
        dot_group = VGroup(*[Dot(p, radius=0.02, color=GREEN) for p in points])
        self.play(Transform(frontier_line, dot_group))
        self.next_slide()

        drones = [drone_1, drone_2, drone_3]
        arrows = [VGroup(), VGroup(), VGroup()]
        for i, drone in enumerate(drones):
            pos = drone.get_center()
            for p in points:
                arrows[i].add(Arrow(
                    start=pos, end=p,
                    buff=0.08, stroke_width=1.5, tip_length=0.07, color=RED
                ))

        self.play(Succession(
            FadeIn(arrows[0]),
            Transform(arrows[0], arrows[1].copy()),
            Transform(arrows[0], arrows[2].copy()),
        ))
        self.remove(*arrows)
        self.next_slide()
        self.play(FadeOut(arrows[0]))
        self.next_slide()

        # Record where the camera is now (still zoomed in)
        zoomed_center = self.camera.frame.get_center().copy()

        # frontier_line was Transform'd into dot_group — it IS the green dots on screen
        self._show_split_clustering(
            drone_1=drone_1,
            drone_2=drone_2,
            drone_3=drone_3,
            drone_colors=drone_colors,
            points=points,
            frontier_dots=frontier_line,
            known_area=known_area,
            area_original=Area,
            person=person,
            zoomed_cam_center=zoomed_center,
            split_scale=0.7,   # ← tweak this to taste: 1.0 = original size
        )