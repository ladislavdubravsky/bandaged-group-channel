from manimlib.imports import *


class Sudoku(VGroup):
    CONFIG = {
        "cell_size": 0.8,
        "border_stroke_width": 8,
        "border_stroke_color": "#000000",
        "box_stroke_width": 5,
        "box_stroke_color": "#999999",
        "cell_stroke_width": 0.5,
        "cell_stroke_color": "#000000",
        "fill_opacity": 1,
        "stroke_color": "#000000",
        "fill_color": WHITE,
        "col1": "#F9CDC4",
    }

    def __init__(self, scene, cages=None, signs=None, **kwargs):
        VGroup.__init__(self, **kwargs)
        self.candidates = {}
        self.sums = {}
        self.highlights = []
        self.hld_cells = set()
        self.scene = scene
        self.add_grid(cages, signs)

    def add_grid(self, cages=None, signs=None):
        cell_config = {
            "side_length": self.cell_size,
            "stroke_width": self.cell_stroke_width,
            "stroke_color": self.cell_stroke_color,
            "fill_opacity": 0.4,
            "fill_color": self.fill_color,
        }
        cells = VGroup(
            *[VGroup(*[Square(**cell_config) for _ in range(9)]).arrange(buff=0) for _ in range(9)]
        ).arrange(buff=0, direction=DOWN)
        self.cells = cells
        self.add(cells)

        box_config = {
            "side_length": 3 * self.cell_size,
            "stroke_width": self.box_stroke_width,
            "stroke_color": self.box_stroke_color,
        }
        boxes = VGroup(
            *[VGroup(*[Square(**box_config) for _ in range(3)]).arrange(buff=0) for _ in range(3)]
        ).arrange(buff=0, direction=DOWN)
        self.boxes = boxes
        self.add(boxes)

        border = Square(
            side_length=9 * self.cell_size,
            stroke_width=self.border_stroke_width,
            stroke_color=self.border_stroke_color
        )
        self.border = border
        self.add(border)

        buff = 0.09
        self.cages = []
        for cage_border_coords in cages:
            cage_border, cage_cells = [], []
            for u, v in zip(cage_border_coords, cage_border_coords[1:] + [cage_border_coords[0]]):
                line = DashedLine(
                    self.cells[u[0]][u[1]].get_corner(u[2]) - buff * u[2],
                    self.cells[v[0]][v[1]].get_corner(v[2]) - buff * v[2],
                    stroke_color="#000000",
                    stroke_width=2,
                    dash_length=0.3,
                    positive_space_ratio=0.75,
                )
                cage_border.append(line)
                rstart, rend = min(u[0], v[0]), max(u[0], v[0]) + 1
                cstart, cend = min(u[1], v[1]), max(u[1], v[1]) + 1
                cage_cells.extend([(r, c) for r in range(rstart, rend) for c in range(cstart, cend)])
            cage = VGroup(*cage_border)
            self.add(cage)
            self.cages.append(set(cage_cells))

        for sign_coords in signs:
            r1, c1, r2, c2, s = sign_coords
            sign = TexMobject(r"\bm{" + s + r"}")
            pos = (self.cells[r1][c1].get_center() + self.cells[r2][c2].get_center()) / 2
            sign.move_to(pos).set_color(r"#0000FF")
            if r1 != r2:
                sign.rotate(PI / 2)
            br = BackgroundRectangle(sign, fill_opacity=1, color=WHITE)
            sign = VGroup(br, sign)
            self.add(sign)

    def cage_at(self, r, c):
        return [cage for cage in self.cages if (r, c) in cage][0]

    def add_text(self, r, c, text, aligned_edge=ORIGIN, color="#0000FF", scale=1., wait=0,
                 br_color=WHITE, br_opacity=0):
        text = [text] if isinstance(text, str) else text
        t = VGroup(*[TexMobject(t).set_color(color).scale(scale) for t in text])
        t.arrange(direction=DOWN, buff=SMALL_BUFF)
        t.move_to(self.cells[r][c], aligned_edge=aligned_edge)
        bt = BackgroundRectangle(t, fill_opacity=br_opacity, color=br_color, buff=0.01)
        t = VGroup(bt, t)
        self.scene.add_foreground_mobject(t)
        if wait:
            self.scene.wait(wait)
        return t

    def add_sum(self, r, c, sum, scale=0.6):
        self.clear_sum(r, c)
        s = self.add_text(r, c, r"\textbf{" + sum + "}", aligned_edge=UL, color="#000000",
                          scale=scale, br_color=YELLOW, br_opacity=1)
        self.sums[(r, c)] = s
        self.scene.wait()

    def set_candidates(self, r, c, *candidates, cage=False):
        cells = [(r, c)] if not cage else self.cage_at(r, c)
        scale = {1: 1, 2: 0.7, 3: 0.6, 4: 0.4, 5: 0.4}[max(len(c) for c in candidates)]
        for (ri, ci) in cells:
            self.clear_text(ri, ci)
            t = self.add_text(ri, ci, candidates, scale=scale)
            self.candidates[(ri, ci)] = t
        self.scene.wait()

    def clear_text(self, r, c):
        self.scene.remove(self.candidates.get((r, c), None))

    def clear_sum(self, r, c):
        self.scene.remove(self.sums.get((r, c), None))

    def hl_cell(self, r, c, **kwargs):
        self.hl_cells([(r, c)], **kwargs)

    def hl_cells(self, coords, color=1, overdraw=False, opacity=0.4):
        hl = VGroup(*[
            deepcopy(self.cells[r][c]) for (r, c) in coords if (r, c) not in self.hld_cells or overdraw
        ])
        self.hld_cells.update(set(coords))
        hl.set_fill(color={1: self.col1, 2: self.col2}[color])
        hl.set_opacity(opacity)
        self.scene.bring_to_back(hl)
        self.highlights.append(hl)
        self.scene.wait()
        return hl

    def hl_row(self, r, **kwargs):
        return self.hl_cells([(r, c) for c in range(9)], **kwargs)

    def hl_col(self, c, **kwargs):
        return self.hl_cells([(r, c) for r in range(9)], **kwargs)

    def hl_cage(self, r, c, **kwargs):
        return self.hl_cells(self.cage_at(r, c), **kwargs)

    def hl_box(self, n):
        box = deepcopy(self.boxes[(n - 1) // 3][(n - 1) % 3])
        box.set_style(stroke_color=ORANGE, stroke_width=8)
        self.scene.add(box)
        self.highlights.append(box)
        self.scene.wait()
        return box

    def clear_highlights(self):
        self.scene.remove(*self.highlights)
        self.highlights.clear()
        self.hld_cells.clear()


class SudokuScene(MovingCameraScene):
    """ python -m manim sudoku.py SudokuScene -pl """

    def construct(self):
        # definition
        cages = [
            [(0, 1, UL), (0, 3, UR), (0, 3, DR), (0, 1, DR), (1, 1, DR), (1, 0, DR), (4, 0, DR), (4, 0, DL), (1, 0, UL), (1, 1, UL)],
            [(1, 2, UL), (1, 2, UR), (2, 2, UR), (2, 3, UR), (3, 3, DR), (3, 3, DL), (2, 3, DL), (2, 1, DR), (3, 1, DR), (3, 1, DL), (2, 1, UL), (2, 2, UL)],
            [(0, 4, UL), (0, 6, UR), (0, 6, DR), (0, 4, DL)],
            [(1, 3, UL), (1, 4, UR), (1, 4, DR), (1, 3, DL)],
            [(0, 7, UL), (0, 8, UR), (1, 8, DR), (1, 8, DL), (0, 8, DL), (0, 7, DL)],
            [(1, 6, UL), (1, 6, UR), (2, 6, UR), (2, 8, UR), (2, 8, DR), (2, 6, DL)],
            [(3, 2, UL), (3, 2, UR), (5, 2, DR), (5, 0, DR), (6, 0, DR), (6, 0, DL), (5, 0, UL), (5, 1, UL), (4, 1, UL), (4, 2, UL)],
            [(2, 5, UL), (2, 5, UR), (4, 5, DR), (4, 4, DR), (5, 4, DR), (5, 3, DR), (6, 3, DR), (6, 3, DL), (4, 3, UL), (4, 4, UL), (3, 4, UL), (3, 5, UL)],
            [(3, 6, UL), (3, 8, UR), (3, 8, DR), (3, 7, DR), (4, 7, DR), (4, 6, DR), (5, 6, DR), (5, 5, DL), (5, 5, UL), (5, 6, UL)],
            [(4, 8, UL), (4, 8, UR), (5, 8, DR), (5, 7, DR), (6, 7, DR), (6, 7, DL), (5, 7, UL), (5, 8, UL)],
            [(7, 1, UL), (7, 1, UR), (8, 1, UR), (8, 2, UR), (8, 2, DR), (8, 1, DL)],
            [(6, 1, UL), (6, 2, UR), (7, 2, UR), (7, 4, UR), (7, 4, DR), (7, 2, DL), (6, 2, DL), (6, 1, DL)],
            [(6, 4, UL), (6, 6, UR), (6, 6, DR), (6, 5, DR), (7, 5, DR), (7, 5, DL), (6, 5, DL), (6, 4, DL)],
            [(8, 3, UL), (8, 5, UR), (8, 5, DR), (8, 3, DL)],
            [(7, 6, UL), (7, 7, UR), (8, 7, DR), (8, 6, DL)],
            [(6, 8, UL), (6, 8, UR), (7, 8, DR), (7, 8, DL)],
        ]
        signs = [(0, 4, 1, 4, r"="), (0, 6, 0, 7, r"<"), (2, 6, 3, 6, r"<"), (5, 5, 6, 5, r">"), (6, 6, 6, 7, r"<"), (7, 7, 7, 8, r"<"), (7, 1, 7, 2, r"<"), (8, 5, 8, 6, r"=")]
        s = Sudoku(self, cages=cages, signs=signs)
        self.add(s)
        attribution = TexMobject(r"\text{\copyright Miyuki Misawa}")
        attribution.scale(0.5).to_edge(DOWN, buff=SMALL_BUFF / 2).set_color("#000000")
        self.add(attribution)
        self.wait()

        # thumbnail etc.
        # self.wait(2)
        # self.camera_frame.save_state()
        # self.play(
        #     self.camera_frame.shift, (2, -1.5, 0),
        #     self.camera_frame.set_width, 4,
        #     run_time=2
        # )
        # self.wait()
        # self.play(
        #     self.camera_frame.restore,
        #     run_time=2
        # )
        # self.wait(6)
        # return

        # self.add_sound("audio")

        # display rules
        rules = VGroup(
            TexMobject(r"\text{1. Normal sudoku rules apply}"),
            TexMobject(r"\text{2. No cage can contain a number more than once}"),
            TexMobject(r"\text{3. Sums of numbers in cages obey the indicated relationships}")
        ).arrange(direction=DOWN)
        br = BackgroundRectangle(rules, fill_opacity=0.75, color="#000000", buff=MED_SMALL_BUFF)
        br.stretch(1.2, 0)
        rules = VGroup(br, rules)
        self.add(rules)
        self.play(FadeIn(rules))
        self.wait()
        self.remove(rules)
        self.play(FadeOut(rules))

        # rule examples
        s.hl_cage(4, 6)
        s.hl_cage(6, 6)
        s.clear_highlights()
        s.hl_cage(0, 4)
        s.hl_cage(1, 4)
        s.clear_highlights()

        ##### SOLUTION
        # A B C
        abc_style = dict(br_color=YELLOW, br_opacity=1, aligned_edge=UR, scale=0.75, wait=1, color="#0000FF")
        s.hl_cell(6, 0)
        s.add_text(6, 0, r"\textbf{A}", **abc_style)
        s.hl_col(0)
        s.hl_cage(4, 1)
        hl = s.hl_box(4)
        s.add_text(3, 1, r"\textbf{A}", **abc_style)
        self.remove(hl)
        s.hl_col(1)
        s.hl_cage(1, 2)
        s.hl_box(1)
        s.add_text(0, 2, r"\textbf{A}", **abc_style)
        s.clear_highlights()

        s.hl_cells([(2, 5)])
        s.add_text(2, 5, r"\textbf{B}", **abc_style)
        s.hl_col(5)
        s.hl_cage(2, 5)
        s.hl_box(5)
        s.add_text(3, 3, r"\textbf{B}", **abc_style)
        s.clear_highlights()

        s.add_text(6, 3, r"\textbf{C}", **abc_style)
        s.add_text(5, 5, r"\textbf{C}", **abc_style)
        s.add_text(4, 8, r"\textbf{C}", **abc_style)

        # 28-29-30 cages
        s.hl_cage(3, 6)
        s.add_sum(3, 6, r"28+")
        s.set_candidates(3, 6, "1234", "567", cage=True)
        s.hl_cage(6, 4)
        s.add_sum(6, 4, r"29+")
        s.hl_cage(4, 8)
        s.add_sum(4, 8, r"30+")
        s.set_candidates(4, 8, "6789", cage=True)
        s.add_sum(4, 8, "30")
        s.add_sum(6, 4, "29")
        s.add_sum(3, 6, "28")
        s.set_candidates(6, 4, "5789", cage=True)
        s.clear_highlights()
        s.add_sum(1, 6, r"29/30", scale=0.4)
        s.set_candidates(1, 6, "56789", cage=True)

        # C is 67
        s.hl_cells([(5, 5), (4, 8)])
        c_values = TexMobject(r"\textbf{C = 67}").next_to(s.border, direction=RIGHT, buff=LARGE_BUFF)
        c_values.shift(UP).set_color("#000000")
        self.add(c_values)

        s.set_candidates(6, 3, "67")
        s.set_candidates(5, 5, "67")
        s.set_candidates(4, 8, "67")
        s.clear_highlights()

        # 28+30 cell
        s.hl_cage(3, 6)
        s.hl_cage(4, 8)
        hl = s.hl_box(6)
        s.clear_highlights()
        s.hl_cells([(5, 5), (6, 7)])
        s.set_candidates(6, 7, "67")
        s.clear_highlights()
        s.remove(hl)

        # 30 cell work
        s.hl_cells([(6, 7), (4, 8)])
        s.set_candidates(5, 7, "89")
        s.set_candidates(5, 8, "89")
        s.clear_highlights()

        # 29 cell work
        s.hl_cells([(6, 3), (6, 7)])
        s.hl_cage(6, 4, color=2)
        s.set_candidates(7, 5, "7")
        s.set_candidates(6, 4, "589")
        s.set_candidates(6, 5, "589")
        s.set_candidates(6, 6, "589")
        s.clear_highlights()

        # C is 6
        s.set_candidates(5, 5, "6")
        s.set_candidates(6, 3, "6")
        s.set_candidates(4, 8, "6")
        self.remove(c_values)
        c_values = TexMobject(r"\textbf{C = 6}").set_color("#FF0000").next_to(s.border, direction=RIGHT, buff=LARGE_BUFF)
        self.add(c_values)
        s.set_candidates(6, 7, "7")

        # box 9 cage sums determined
        s.hl_row(6)
        s.hl_col(8)
        s.hl_cage(7, 7, color=2)
        s.add_sum(7, 6, r"12+")
        s.add_sum(6, 8, r"13+")
        s.hl_cell(6, 8, overdraw=True, color=2, opacity=1)
        s.set_candidates(6, 8, "4")
        s.set_candidates(7, 8, "9")
        s.add_sum(6, 8, "13")
        s.add_sum(7, 6, "12")
        s.set_candidates(7, 6, "1236", cage=True)
        s.set_candidates(6, 6, "58")
        s.set_candidates(8, 8, "58")
        s.clear_highlights()
        s.set_candidates(5, 7, "9")
        s.set_candidates(5, 8, "8")
        s.set_candidates(8, 8, "5")
        s.set_candidates(6, 6, "8")
        s.set_candidates(6, 4, "59")
        s.set_candidates(6, 5, "59")
        s.set_candidates(2, 8, "7")
        s.hl_cage(2, 7)
        s.set_candidates(2, 7, "8")
        s.set_candidates(0, 8, "123")
        s.set_candidates(1, 8, "123")
        s.set_candidates(3, 8, "123")
        s.set_candidates(1, 6, "569")
        s.set_candidates(2, 6, "569")
        s.clear_highlights()

        # A is 123
        s.set_candidates(6, 0, "123")
        s.set_candidates(6, 1, "123")
        s.set_candidates(6, 2, "123")
        a_values = TexMobject(r"\textbf{A = 123}").next_to(c_values, direction=DOWN).set_color("#000000")
        self.add(a_values)


        # b_values = TexMobject(r"\textbf{B = 3459}").next_to(a_values, direction=DOWN).set_color("#000000")
        # self.add(b_values)
        self.wait()
