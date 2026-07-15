from manim import *
from manim_slides import Slide
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict


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
    #  Utility helpers (unchanged from original)
    # ──────────────────────────────────────────────────

    def sample_perimeter(self, vmobj: VMobject, step: float = 0.25) -> np.ndarray:
        points = vmobj.get_points()
        if len(points) < 2:
            return points.copy()
        diffs   = np.diff(points, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        cumlen   = np.concatenate([[0], np.cumsum(seg_lens)])
        total_len = cumlen[-1]
        positions = np.arange(0, total_len, step)
        positions[-1] = min(positions[-1], total_len)
        sample_points = []
        for t in positions:
            idx = np.searchsorted(cumlen, t, side="right") - 1
            idx = min(idx, len(seg_lens) - 1)
            seg_len = seg_lens[idx]
            t_in_seg = t - cumlen[idx]
            alpha = t_in_seg / seg_len if seg_len > 0 else 0.0
            p1, p2 = points[idx], points[idx + 1]
            sample_points.append(p1 + alpha * (p2 - p1))
        return np.array(sample_points)

    def set_submobject_fill_opacity(self, mobject: Mobject, opacity: float):
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
            fill_color=WHITE, fill_opacity=0.5, stroke_width=0,
            z_index=Area.z_index
        )

    def drone_generator(self, Area, DRONE_COLOR=RED):
        drone = SVGMobject("DPGMM/Drone.svg").scale(0.1).set_color(DRONE_COLOR)
        drone.z_index = 2
        drone.move_to(Area.get_edge_center(DOWN + LEFT)).shift(0.3 * UP + 0.3 * RIGHT)
        FoV = Circle(radius=0.5, z_index=1).move_to(drone.get_center())
        return drone, FoV

    def points_bounded_by_area(self, points: np.ndarray, area: VMobject) -> np.ndarray:
        corners = area.get_vertices()
        xs, ys  = corners[:, 0], corners[:, 1]
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        x, y = points[:, 0], points[:, 1]
        mask = (x >= x_min + 0.1) & (x <= x_max - 0.1) & (y >= y_min + 0.1) & (y <= y_max - 0.1)
        return points[mask]

    # ──────────────────────────────────────────────────
    #  Clustering helpers
    # ──────────────────────────────────────────────────

    def _build_kmeans_dots(self, points, drone_positions, drone_colors, dot_radius=0.02):
        """Run KMeans(k=3); cluster color = nearest drone's color."""
        k = 3
        X = points[:, :2]
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels  = kmeans.fit_predict(X)
        centers = kmeans.cluster_centers_          # (k, 2) in scene xy

        drone_xy = np.array([d[:2] for d in drone_positions])
        cluster_colors = []
        for c in centers:
            nearest = int(np.argmin(np.linalg.norm(drone_xy - c, axis=1)))
            cluster_colors.append(drone_colors[nearest])

        dots = VGroup(*[
            Dot(pt, radius=dot_radius, color=cluster_colors[lbl])
            for pt, lbl in zip(points, labels)
        ])

        stars = VGroup(*[
            Star(n=5, outer_radius=0.06, color=cluster_colors[ci], fill_opacity=1)
            .move_to([cx, cy, 0])
            for ci, (cx, cy) in enumerate(centers)
        ])

        return dots, stars, cluster_colors, labels, centers

    def _build_dpgmm_dots(self, points, centers, cluster_colors, dot_radius=0.02):
        """
        Soft-assignment coloring: each dot blends the 3 cluster colors
        weighted by inverse distance → smooth gradient at boundaries.
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
    #  Split-screen section
    # ──────────────────────────────────────────────────

    def _show_split_clustering(
        self, *,
        area_original: VMobject,
        known_area: VMobject,
        points: np.ndarray,
        drone_1, drone_2, drone_3,
        drone_colors: list,
        # objects to hide before splitting
        objects_to_hide: list,
    ):
        """
        1. Fade out the original single scene.
        2. Create two side-by-side panels (left=DPGMM, right=KMeans).
        3. Reveal clusters.
        4. Zoom out to show both panels in full.
        """
        scale = 0.8
        area_w   = area_original.width      # 6.5
        area_h   = area_original.height     # 4.5
        # Centre of original area in scene space
        orig_cx  = area_original.get_center()[0]
        orig_cy  = area_original.get_center()[1]

        # ── Spacing: panels placed area_w apart (gap = area_w * 0.15 between them) ──
        gap          = area_w * 0.15
        panel_offset = area_w + gap / 4      # how far each panel shifts from orig centre

        # Absolute x of each panel centre
        x_right =  0   # right panel (KMeans)
        x_left  = -0   # left  panel (DPGMM)
        # Shift vectors relative to the original area's centre
        shift_right = np.array([x_right - orig_cx, 0, 0])
        shift_left  = np.array([x_left  - orig_cx, 0, 0])

        # ── Compute clusters on original-space points ──
        drone_positions = [d.get_center() for d in [drone_1, drone_2, drone_3]]
        km_dots, km_stars, cluster_colors, km_labels, km_centers_xy = \
            self._build_kmeans_dots(points, drone_positions, drone_colors)
        dpgmm_dots = self._build_dpgmm_dots(points, km_centers_xy, cluster_colors)

        # ── Build panel scene objects ──
        # Right panel: KMeans
        area_right  = area_original.copy().shift(shift_right)
        known_right = known_area.copy().shift(shift_right)
        d1r = drone_1.copy().shift(shift_right)
        d2r = drone_2.copy().shift(shift_right)
        d3r = drone_3.copy().shift(shift_right)
        km_dots_right  = km_dots.copy().shift(shift_right)
        km_stars_right = km_stars.copy().shift(shift_right)

        # Left panel: DPGMM
        area_left   = area_original.copy().shift(shift_left)
        known_left  = known_area.copy().shift(shift_left)
        d1l = drone_1.copy().shift(shift_left)
        d2l = drone_2.copy().shift(shift_left)
        d3l = drone_3.copy().shift(shift_left)
        dpgmm_dots_left = dpgmm_dots.copy().shift(shift_left)

        # ── Divider: vertical dashed line between panels, full area height ──
        # The divider sits at x = 0 (midpoint between left and right panels)
        divider_x  = 0.0
        divider_y  = orig_cy
        divider_half_h = area_h * 0.6
        divider = DashedLine(
            start=[divider_x, divider_y + divider_half_h, 0],
            end  =[divider_x, divider_y - divider_half_h, 0],
            dash_length=0.15,
            dashed_ratio=0.5,
            color=LIGHT_GREY,
            stroke_width=2.5,
            z_index=10,
        )

        # ── Labels: sit just above each area rectangle ──
        label_top_y = orig_cy + area_h / 2 + 0.35
        label_kmeans = Text("Hard Clustering", font_size=28, color=WHITE, weight=BOLD) \
            .move_to([x_right, label_top_y, 0])
        label_dpgmm  = Text("Soft Clustering", font_size=28, color=WHITE, weight=BOLD) \
            .move_to([x_left,  label_top_y, 0])

        # ── Camera target for the split view ──
        # Centre between the two panels, wide enough to show both + margins
        split_cam_cx    = (x_left + x_right) / 2   # = 0.0
        split_cam_cy    = orig_cy
        split_cam_width = abs(x_right - x_left) + area_w + 2.0   # both panels + padding

        # ── Step 0: Fade out original objects, bring in both panels simultaneously ──
        self.play(
            *[FadeOut(obj) for obj in objects_to_hide],
            # Right panel
            FadeIn(area_right), FadeIn(known_right),
            FadeIn(d1r), FadeIn(d2r), FadeIn(d3r),
            # Left panel
            FadeIn(area_left),  FadeIn(known_left),
            FadeIn(d1l), FadeIn(d2l), FadeIn(d3l),
            # Divider + labels
            Create(divider),
            FadeIn(label_kmeans),
            FadeIn(label_dpgmm),
            # Camera: zoom out to show both panels
            self.camera.frame.animate
                .move_to([split_cam_cx, split_cam_cy, 0])
                .set(width=split_cam_width),
            run_time=1.8,
        )
        self.next_slide()

        # # ── Step 1: Reveal KMeans clusters (right) ──
        # self.play(FadeIn(km_dots_right), run_time=0.9)
        # self.play(
        #     LaggedStart(*[GrowFromCenter(s) for s in km_stars_right], lag_ratio=0.3),
        #     run_time=0.7,
        # )
        # self.next_slide()

        # # ── Step 2: Reveal DPGMM soft clusters (left) with stagger for gradient feel ──
        # self.play(
        #     LaggedStart(
        #         *[FadeIn(d) for d in dpgmm_dots_left],
        #         lag_ratio=0.008,
        #     ),
        #     run_time=1.8,
        # )
        # self.next_slide()

        # # ── Step 3: Zoom out to default frame (show full rectangles + context) ──
        # full_width  = split_cam_width + 2.0
        # self.play(
        #     self.camera.frame.animate
        #         .move_to([split_cam_cx, split_cam_cy, 0])
        #         .set(width=full_width),
        #     run_time=1.5,
        # )
        # self.next_slide()

    # ──────────────────────────────────────────────────
    #  Main construct / show_introduction
    # ──────────────────────────────────────────────────

    
    def construct(self):
        # self.show_title()
        self.show_introduction()

    def show_title(self):
        title = Text("“That’s Probably My Job”:\nTeaching Robots to Softly Share Tasks").move_to(ORIGIN)
        authors = Text(
            "John Lewis Devassy, Meysam Basiri, "
            "Mário A. T. Figueiredo, and Pedro U. Lima"
        ).next_to(title, DOWN, buff=0.5).scale(0.4).fade(0.75)
        self.play(Write(title), FadeIn(authors))
        self.next_slide()
        self.play(FadeOut(title), FadeOut(authors))
    
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
        self.remove(title)
        self.next_slide()

        known_area = Union(
            drone_1_FoV_AoI, drone_2_FoV_AoI, drone_3_FoV_AoI, fill_opacity=0.3
        )
        known_area.set_fill(opacity=0.3)
        self.play(FadeIn(known_area))

        continous_frontier = known_area.copy() \
            .set_fill(opacity=0).set_stroke(color=GREEN, width=2.0)
        self.play(Create(continous_frontier))
        self.next_slide()

        points    = self.sample_perimeter(known_area, step=0.1)
        points    = self.points_bounded_by_area(points, Area)
        dot_group = VGroup(*[Dot(p, radius=0.02, color=GREEN) for p in points])
        self.play(Transform(continous_frontier, dot_group))
        self.next_slide()

        drones = VGroup(drone_1, drone_2, drone_3)
        arrows = [VGroup(), VGroup(), VGroup()]
        for iter, drone in enumerate(drones):
            drone_pos = drone.get_center()
            for p in points:
                arrows[iter].add(Arrow(
                    start=drone_pos, end=p,
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

        # ── Fade the frontier dots out (they will reappear clustered per panel) ──
        self.play(FadeOut(continous_frontier))

        # ── Hand off to split-screen section ──
        # Everything currently on screen that belongs to the single-area view:
        original_objects = [Area, known_area, person, drone_1, drone_2, drone_3]

        self._show_split_clustering(
            area_original=Area,
            known_area=known_area,
            points=points,
            drone_1=drone_1,
            drone_2=drone_2,
            drone_3=drone_3,
            drone_colors=drone_colors,
            objects_to_hide=original_objects,
        )