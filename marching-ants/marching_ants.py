#!/usr/bin/env python
import itertools as it
import functools
from operator import itemgetter
from typing import List, Tuple, Dict

import networkx as nx

from manimlib.imports import *

GRAPH = "graph"
EDGE_HIGHLIGHTS = "edge_hls"
FACE_HIGHLIGHTS = "face_hls"
EDGE_MARKS = "edge_marks"
VERTEX_MARKS = "vertex_marks"
KEYS = [GRAPH, EDGE_HIGHLIGHTS, FACE_HIGHLIGHTS, EDGE_MARKS, VERTEX_MARKS]


class MyGraph:
    scene: Scene  # scene where everything will be drawn
    drawings: Dict[bool, Dict[str, List]]  # all drawings, 2d and 3d version

    vertices: List[np.array]     # coordinates with 3rd value zero
    vertices_3d: List[np.array]  # coordinates in same order as vertices
    edges: List[Tuple[int, int]]      # indices into vertices(_3d)
    faces: List[Tuple[int, ...]]      # indices into vertices(_3d)

    def __init__(self, scene, is_3d=False):
        self.vertices = self.vertices_3d = self.edges = self.faces = []
        self.scene = scene
        self.drawings = {
            False: {k: [] for k in KEYS},  # 2d drawings
            True: {k: [] for k in KEYS},  # 3d drawings
        }
        self.is_3d = is_3d

    def erase(self, *keys, animate=False):
        for key in keys:
            self.scene.remove(*self.drawings[0][key])
            self.scene.remove(*self.drawings[1][key])
            if animate:
                self.scene.play(AnimationGroup(
                    *[FadeOut(o) for o in self.drawings[self.is_3d][key]]
                ))
            self.drawings[0][key].clear()
            self.drawings[1][key].clear()
    def erase_graph(self, **kwargs): self.erase(GRAPH, **kwargs)
    def erase_edge_highlights(self, **kwargs): self.erase(EDGE_HIGHLIGHTS, **kwargs)
    def erase_face_highlights(self, **kwargs): self.erase(FACE_HIGHLIGHTS, **kwargs)
    def erase_edge_marks(self, **kwargs): self.erase(EDGE_MARKS, **kwargs)
    def erase_vertex_marks(self, **kwargs): self.erase(VERTEX_MARKS, **kwargs)
    def erase_all(self, **kwargs): self.erase(*KEYS[1:], **kwargs)

    @staticmethod
    def do_for_2d_and_3d(func):
        # TODO decorator
        pass

    def anim_2d_3d(self, **kwargs):
        kwargs = {"rate_func": linear, **kwargs}
        transforms = []
        for key in KEYS:
            obj1 = self.drawings[self.is_3d][key]
            obj2 = self.drawings[not self.is_3d][key]
            if obj1 and obj2:
                obj1, obj2 = Group(*obj1), Group(*obj2)
                transforms.append(Transform(obj1, obj2, **kwargs))
        self.scene.play(AnimationGroup(*transforms))
        self.is_3d = not self.is_3d

    def anim_rotating(self, **kwargs):
        anims = []
        for key in KEYS:
            obj = self.drawings[self.is_3d][key]
            if obj:
                obj = Group(*obj)
                anims.append(Rotating(obj, **kwargs))
        self.scene.play(AnimationGroup(*anims))

    def draw_graph(self, animate=False, bg=set(), **kwargs):
        kwargs = {"color": "#bbbbbb", **kwargs}
        self.erase(GRAPH, animate=False)
        for vs, is_3d in (self.vertices, False), (self.vertices_3d, True):
            def fo(i, j): return 0.2 if i in bg or j in bg else 1
            edges = [Line(vs[i], vs[j], **kwargs, stroke_opacity=fo(i, j)) for i, j in self.edges]

            self.drawings[is_3d][GRAPH] = edges
            if is_3d == self.is_3d:
                self.scene.add(*edges)
                self.scene.bring_to_back(*edges)
                if animate:
                    self.scene.play(ShowCreation(Group(*edges)))

    def highlight_edges(self, edges, animate=True, shift=ORIGIN, foreground=False, bg=set(), **kwargs):
        kwargs = {"color": YELLOW, **kwargs}
        if isinstance(edges, dict):
            edges = edges.items()
        for vs, is_3d in (self.vertices, False), (self.vertices_3d, True):
            def fo(i, j):
                return 0.2 if i in bg or j in bg else 1
            highlight = [Line(vs[i] + shift, vs[j] + shift, stroke_opacity=fo(i, j), **kwargs) for i, j in edges]

            self.drawings[is_3d][EDGE_HIGHLIGHTS].extend(highlight)
            if is_3d == self.is_3d:
                if not foreground:
                    self.scene.add(*highlight)
                else:
                    self.scene.add_foreground_mobjects(*highlight)
                if hasattr(self.scene, "ants"):
                    reset_ants(self.scene)
                anim = [ShowCreation(h) for h in highlight]
                if animate:
                    self.scene.play(AnimationGroup(*anim))
                else:
                    return AnimationGroup(*anim)


    def circle_ant(self, vertex, run_time=1.5):
        vs = self.vertices_3d if self.is_3d else self.vertices
        self.scene.play(ShowCreationThenDestruction(
            Circle(
                arc_center=vs[vertex], radius=0.7, stroke_width=6, color=YELLOW
            ),
            run_time=run_time
        ))

    def get_boundary(self, faces):
        edges = [
            e
            for f in map(lambda f: list(zip(f, f[1:] + (f[0],))), faces)
            for e in f
        ]
        edges = {e for e in edges if edges.count(e) + edges.count((e[1], e[0])) == 1}
        res = [edges.pop()]
        while edges:
            for e in edges:
                if e[0] == res[-1][1]:
                    edges.remove(e)
                    res.append(e)
                    break
                if e[1] == res[-1][1]:
                    edges.remove(e)
                    res.append((e[1], e[0]))
                    break
        return res


    def highlight_faces(self, faces, animate=True, rm_inedges=True, reset=True, **kwargs):
        kwargs = {"color": YELLOW, "fill_color": BLUE, "fill_opacity": 0.5, **kwargs}
        for vs, is_3d in (self.vertices, False), (self.vertices_3d, True):
            highlight = [Polygon(*[vs[i] for i in f], **kwargs) for f in faces]

            # draw over interior (shared) edges with fill_color to create one region
            if rm_inedges:
                edges = [
                    tuple(sorted(e))
                    for f in map(lambda f: list(zip(f, f[1:] + (f[0], ))), faces)
                    for e in f
                ]
                for e in filter(lambda e: edges.count(e) > 1, edges):
                    highlight.append(Line(vs[e[0]], vs[e[1]],
                                          color=kwargs["fill_color"],
                                          fill_opacity=kwargs["fill_opacity"]))

            self.drawings[is_3d][FACE_HIGHLIGHTS].extend(highlight)
            if is_3d == self.is_3d:
                self.scene.add(*highlight)
                if hasattr(self.scene, "ants") and reset:
                    reset_ants(self.scene)
                if animate:
                    self.scene.play(AnimationGroup(*[ShowCreation(h) for h in highlight]))

    def cross_edges(self, edge_ids, animate=True, **kwargs):
        kwargs = {"color": RED, **kwargs}
        size = 0.1
        for vs, is_3d in (self.vertices, False), (self.vertices_3d, True):
            crosses = []
            for edge_id in edge_ids:
                edge = self.edges[edge_id]
                center = (vs[edge[0]] + vs[edge[1]]) / 2
                crosses.append(VGroup(
                    Line(center - np.array([size, size, 0]), center + np.array([size, size, 0]), **kwargs),
                    Line(center - np.array([-size, size, 0]), center + np.array([-size, size, 0]), **kwargs),
                ))
                # t = TextMobject(str(edge_id))
                # t.move_to(center)
                # self.scene.add(t)

            self.drawings[is_3d][EDGE_MARKS].extend(crosses)
            if is_3d == self.is_3d:
                self.scene.add(*crosses)
                if animate:
                    self.scene.play(
                        AnimationGroup(*[ShowCreation(h) for h in crosses], lag_ratio=0.2)
                    )

    def circle_vertices(self, vertex_ids, animate=True, **kwargs):
        kwargs = {"color": BLUE, **kwargs}
        for vs, is_3d in (self.vertices, False), (self.vertices_3d, True):
            circles = []
            for vertex_id in vertex_ids:
                circle = Circle(radius=0.1, **kwargs)
                circle.move_to(vs[vertex_id])
                circles.append(circle)

            self.drawings[is_3d][VERTEX_MARKS].extend(circles)
            if is_3d == self.is_3d:
                self.scene.add(*circles)
                if animate:
                    self.scene.play(
                        AnimationGroup(*[ShowCreation(h) for h in circles], lag_ratio=0.2)
                    )

    def add_arrows(self, paths, size=1.4, **kwargs):
        """ Only 2D """
        arrows = []
        if isinstance(paths, dict):
            paths = paths.items()
        for i, j in paths:
            u, v = self.vertices[i], self.vertices[j]
            dir = normalize(v - u)
            arr = Arrow(u, u + size * dir, **kwargs)
            arrows.append(arr)
        self.scene.add(*arrows)
        if self.scene.ants:
            reset_ants(self.scene)
        return arrows

    def find_cycle_covers(self):
        vertices = set(range(len(self.vertices)))
        faces = list(map(lambda f: list(zip(f, f[1:] + (f[0], ))), self.faces))
        faces = list(map(lambda f: list(map(lambda x: tuple(sorted(x)), f)), faces))

        covers = set()
        total = 2 ** len(faces)
        for n, cfg in enumerate(it.product([0, 1], repeat=len(faces))):
            if n % 1000 == 0:
                print(f"Finding cycle covers... {100 * n / total}%")
            cfg_faces = [f for i, f in enumerate(faces) if cfg[i]]
            cfg_edges = set()
            for f in cfg_faces:
                for e in f:
                    if e in cfg_edges:
                        cfg_edges.remove(e)
                    else:
                        cfg_edges.add(e)
            if vertices == set(v for e in cfg_edges for v in e):
                covers.add(frozenset(cfg_edges))

        return list(covers)


class MyTriangleGraph(MyGraph):
    def __init__(self, scene, **kwargs):
        super().__init__(scene, **kwargs)
        self.vertices = [(-3, -2, 0), (3, -1, 0), (1, 2, 0)]
        self.vertices = [np.array(v) for v in self.vertices]
        self.vertices_3d = self.vertices
        self.edges = [(0, 1), (1, 2), (2, 0)]


class MyCubeGraph(MyGraph):
    def __init__(self, scene, **kwargs):
        super().__init__(scene, **kwargs)
        self.vertices = [(-3, -3, 0), (-3, 1, 0), (1, -3, 0), (1, 1, 0),
                         (3, 3, 0), (3, -1, 0), (-1, 3, 0), (-1, -1, 0)]
        self.vertices = [np.array(v) for v in self.vertices]

        self.vertices_3d = [(-1, -1, -1), (-1, 1, -1), (1, -1, -1), (1, 1, -1),
                            (1, 1, 1), (1, -1, 1), (-1, 1, 1), (-1, -1, 1)]
        self.vertices_3d = [1.5 * np.array(v) for v in self.vertices_3d]

        self.edges = [
            (0, 1), (0, 2), (3, 1), (3, 2),
            (4, 5), (4, 6), (7, 5), (7, 6),
            (0, 7), (1, 6), (2, 5), (3, 4),
        ]
        self.faces = [
            (0, 1, 3, 2), (0, 1, 6, 7), (1, 3, 4, 6),
            (4, 5, 7, 6), (0, 2, 5, 7), (2, 3, 4, 5),
        ]


class MyDodecahedronGraph(MyGraph):
    def __init__(self, scene, **kwargs):
        super().__init__(scene, **kwargs)
        self.vertices = [
            (-0.302, 0.416, 0), # 0
            (0.302, 0.416, 0), # 1
            (0.489, -0.159, 0), # 2
            (0., -0.514, 0), # 3
            (-0.489, -0.159, 0), # 4
            (-0.7, 0.965, 0), # 5
            (0.7, 0.965, 0), # 6
            (1.133, -0.369, 0), # 7
            (0., -1.192, 0), # 8
            (-1.134, -0.368, 0), # 9
            (1.548, 0.503, 0), # 10
            (0., 1.628, 0), # 11
            (-1.548, 0.503, 0), # 12
            (-0.957, -1.317, 0), # 13
            (0.957, -1.317, 0), # 14
            (2.345, 0.762, 0), # 15
            (0., 2.466, 0), # 16
            (-2.345, 0.762, 0), # 17
            (-1.45, -1.995, 0), # 18
            (1.449, -1.995, 0), # 19
        ]
        self.vertices = [1.2 * np.array(v) for v in self.vertices]

        phi = 1.618
        self.vertices_3d = [
            (0, -1 / phi, phi), # 0
            (1, -1, 1), # 1
            (1 / phi, -phi, 0), # 2
            (-1 / phi, -phi, 0), # 3
            (-1, -1, 1), # 4
            (0, 1 / phi, phi), # 5
            (phi, 0 , 1 / phi), # 6
            (1, -1, -1), # 7
            (-1, -1, -1), # 8
            (-phi, 0, 1 / phi), # 9
            (phi, 0, -1 / phi), # 10
            (1, 1, 1), # 11
            (-1, 1, 1), # 12
            (-phi, 0, -1 / phi), # 13
            (0, -1 / phi, -phi), # 14
            (1, 1, -1), # 15
            (1 / phi, phi, 0), # 16
            (-1 / phi, phi, 0), # 17
            (-1, 1, -1), # 18
            (0, 1/ phi, -phi), # 19
        ]
        self.vertices_3d = [1.5 * np.array(v) for v in self.vertices_3d]

        self.faces = [
            (0, 1, 2, 3, 4), (0, 1, 6, 11, 5), (0, 5, 12, 9, 4),
            (4, 9, 13, 8, 3), (2, 3, 8, 14, 7), (1, 2, 7, 10, 6),
            (5, 11, 16, 17, 12), (6, 11, 16, 15, 10), (7, 10, 15, 19, 14),
            (13, 8, 14, 19, 18), (12, 9, 13, 18, 17), (15, 16, 17, 18, 19),
        ]
        edges = [
            tuple(sorted(e))
            for f in map(lambda f: list(zip(f, f[1:] + (f[0], ))), self.faces)
            for e in f
        ]
        seen = set()
        self.edges = [e for e in edges if not (e in seen or seen.add(e))]


def create_ant(point, rotate=0, scale=0.5):
    # ant = Dot(point, radius=0.15, color=RED_E, fill_opacity=1)
    p1 = r"D:\Python\manim\media\videos\marching_ants\mravcisko.png"
    p2 = "/home/ladislav/prj/manim/mravcisko.png"
    path = p1 if os.path.isfile(p1) else p2
    ant = ImageMobject(path)
    ant.rotate(rotate)
    ant.scale(scale)
    ant.move_to(point)
    ant.generate_target()
    return ant


def reset_ants(scene):
    scene.remove(*scene.ants)
    vs = scene.graph.vertices if not scene.graph.is_3d else scene.graph.vertices_3d
    scene.ants = [create_ant(p, **scene.ant_props) for p in vs]
    scene.add(*scene.ants)


def remove_ants(scene):
    scene.play(AnimationGroup(*[FadeOut(o) for o in scene.ants]))
    scene.remove(*scene.ants)
    scene.ants = []


def move_ants(scene, targets, almost=False, foreground=False, ants=None, graph=None, play=True, **kwargs):
    if isinstance(targets, dict):
        targets = targets.items()
    ants = ants if ants else scene.ants
    graph = graph if graph else scene.graph
    for antid, target in targets:
        ant = ants[antid]
        vfrom = graph.vertices[antid] if not graph.is_3d else graph.vertices_3d[antid]
        vto = graph.vertices[target] if not graph.is_3d else graph.vertices_3d[target]
        alpha = 0.8 if almost else 1
        target = (1 - alpha) * vfrom + alpha * vto
        ant.target.move_to(target)
        if foreground:
            scene.add_foreground_mobject(ant)

    if play:
        kwargs = {"run_time": 2, **kwargs}
        scene.play(AnimationGroup(*[MoveToTarget(a) for a in ants]), **kwargs)
    else:
        return [MoveToTarget(a) for a in ants]


class MyCross(Cross):
    CONFIG = {
        "stroke_color": ORANGE,
        "stroke_width": 13,
    }


def intro(scene, headline):
    text = TextMobject(headline)
    scene.add(text)
    scene.play(FadeIn(text, run_time=2))
    scene.wait()
    scene.play(FadeOut(text, run_time=2))
    scene.remove(text)


class IntroScene(MovingCameraScene):
    """ python -m manim marching_ants.py IntroScene -pl """
    ant_props = {"fill_opacity": 0.5}

    def construct(self):
        tr = MyTriangleGraph(self)
        tr.vertices = [v / 1.8 + (-4.5, 0.5, 0) for v in tr.vertices]
        tr.draw_graph(animate=True)

        self.wait(1)
        cube = MyCubeGraph(self)
        cube.vertices = [v / 2.4 + (-0.3, 0.5, 0) for v in cube.vertices]
        cube.draw_graph(animate=True, bg={7})

        self.wait(1)
        dod = MyDodecahedronGraph(self, is_3d=True)
        dod.vertices_3d = [rotation_matrix(-PI / 5, X_AXIS).dot(v) for v in dod.vertices_3d]
        dod.vertices_3d = [v / 1.4 + (4, 0.5, 0) for v in dod.vertices_3d]
        dod.draw_graph(animate=True, bg={0, 1, 2, 3, 4})
        self.wait(1)

        ants = [[create_ant(v, scale=0.25) for v in tr.vertices]]
        ants.append([create_ant(v, scale=0.2) for v in cube.vertices])
        ants.append([create_ant(v, scale=0.12) for v in dod.vertices_3d])
        self.add(*ants[0] + ants[1] + ants[2])
        self.play(FadeIn(Group(*ants[0] + ants[1] + ants[2])))
        self.wait(5)

        a1 = move_ants(self, tr.edges, ants=ants[0], graph=tr, play=False)
        a2 = move_ants(self, cube.get_boundary([cube.faces[2]]) + cube.get_boundary([cube.faces[4]]),
                              ants=ants[1], graph=cube, play=False)
        a3 = move_ants(self, dod.get_boundary(itemgetter(2, 4, 5, 7, 10, 11)(dod.faces)),
                              ants=ants[2], graph=dod, play=False)
        anim = AnimationGroup(AnimationGroup(*a1), AnimationGroup(*a2), AnimationGroup(*a3), lag_ratio=0.7)
        self.play(anim)
        self.wait(8)

        t = [TextMobject(s) for s in ("EASY", "HARD", "MIND BENDING")]
        t[1].to_edge(DOWN, buff=1)
        t[1].shift((-0.5, 0, 0))
        t[0].next_to(t[1], direction=LEFT, buff=2.7)
        t[2].next_to(t[1], direction=RIGHT, buff=1.95)
        for text in t:
            self.add(text)
            self.play(ShowCreation(text, run_time=0.5))
        self.wait(5)
        self.play(*[FadeOut(mob) for mob in self.mobjects])


class TriangleScene(MovingCameraScene):
    """ python -m manim marching_ants.py TriangleScene -pl """
    ants: List[Mobject] = []
    ant_props = {}

    def construct(self):
        intro(self, r"TRIANGLE")

        self.graph = MyTriangleGraph(self)
        self.graph.draw_graph()
        reset_ants(self)
        self.wait(2)

        # collision on a vertex
        move_ants(self, {0: 2, 1: 2, 2: 0})
        cross = MyCross(self.ants[1])
        self.add(cross)
        self.wait()
        self.remove(cross)
        reset_ants(self)

        # collision in the middle of an edge
        midpoint = (self.graph.vertices[0] + self.graph.vertices[1]) / 2
        self.ants[0].target.move_to(midpoint)
        self.ants[1].target.move_to(midpoint)
        self.ants[2].target.move_to(self.graph.vertices[1])
        self.play(
            AnimationGroup(*[MoveToTarget(a, rate_func=rush_into) for a in self.ants]),
            run_time=1.5
        )
        cross = MyCross(self.ants[1])
        self.add(cross)
        self.wait()
        self.remove(cross)
        reset_ants(self)

        # show possible ant decisions
        larrows, rarrows, ltexts, rtexts = [], [], [], []
        for e in self.graph.edges:
            u, v = self.graph.vertices[e[0]], self.graph.vertices[e[1]]
            dir = 1.5 * normalize(v - u)
            norm = 0.5 * normalize(np.array([dir[1], -dir[0], 0]))
            arrow = Arrow(u, u + dir, color=YELLOW)
            arrow.next_to(u, direction=dir)
            rarrows.append(arrow)
            text = TextMobject("R")
            text.next_to(arrow, direction=norm)
            rtexts.append(text)
            arrow = Arrow(v, v - dir, color=YELLOW)
            arrow.next_to(v, direction=-dir)
            larrows.append(arrow)
            text = TextMobject("L")
            text.next_to(arrow, direction=norm)
            ltexts.append(text)
        self.add(*larrows + rarrows + ltexts + rtexts)
        reset_ants(self)
        self.wait(3)

        # show probability table and computation
        self.play(self.camera_frame.move_to, (-2.5, 0, 0))
        table = TexMobject(r"""
            \begin{tabular}{ |c|c|c| }
             \hline
             Ant 1 & Ant 2 & Ant 3 \\
             \hline
             L & L & L \\ 
             L & L & R \\ 
             L & R & L \\
             L & R & R \\
             R & L & L \\
             R & L & R \\
             R & R & L \\
             R & R & R \\
             \hline
            \end{tabular}
        """)
        table.move_to((-7, 0, 0))
        table.scale(0.8)
        self.add(table)
        self.play(ShowCreation(table))
        self.wait(3)

        self.remove(*larrows + ltexts)
        move_ants(self, {0: 1, 1: 2, 2: 0})
        self.wait()
        reset_ants(self)

        self.remove(*rarrows + rtexts)
        self.add(*larrows + ltexts)
        move_ants(self, {0: 2, 2: 1, 1: 0})
        self.wait(2)

        tabler = TexMobject(r"""
            \begin{tabular}{ |c|c|c| } 
             \hline
             Ant 1 & Ant 2 & Ant 3 \\
             \hline
             \hline
             L & L & L \\ 
             \hline
             \hline
             L & L & R \\ 
             L & R & L \\
             L & R & R \\
             R & L & L \\
             R & L & R \\
             R & R & L \\
             \hline
             \hline
             R & R & R \\
             \hline
            \end{tabular}
        """)
        tabler.move_to((-7, 0, 0))
        tabler.scale(0.8)
        self.remove(table)
        self.add(tabler)

        prob = TexMobject(r"p = \frac{2}{8}")
        prob.move_to((-3, 2.5, 0))
        self.play(ShowCreation(prob))
        self.wait(2)
        prob2 = TexMobject(r"p = \frac{1}{4}")
        prob2.move_to((-3, 2.5, 0))
        self.play(Transform(prob, prob2))
        self.wait(3)
        self.play(*[FadeOut(mob) for mob in self.mobjects])


class CubeScene(ThreeDScene):
    """ python -m manim marching_ants.py CubeScene -pl """
    ants = []
    ant_props = {"rotate": 0 * DEGREES, "scale": 0.5}

    def construct(self):
        intro(self, r"CUBE")

        # 6561 total options
        self.graph = MyCubeGraph(self, is_3d=False)
        gr = self.graph
        gr.draw_graph()
        reset_ants(self)
        self.wait(2)
        arrows = gr.add_arrows([*gr.edges, *[(j, i) for i, j in gr.edges]], color=GREEN)
        self.wait(3)
        text = TextMobject(r"\begin{align*} &3.3.3.3.3.3.3.3 = \\ &= 3^8 = 6561 \end{align*}")
        text.to_corner(UL)
        self.add(text)
        self.play(ShowCreation(text, rate_func=linear), run_time=2)
        self.wait(2)
        self.play(FadeOut(text))
        self.remove(text)
        self.remove(*arrows)

        # ant displaces ant
        gr.circle_ant(1, run_time=2.5)

        inv = lambda d: [(j, i) for i, j in d][::-1]
        up = gr.get_boundary([gr.faces[2]])
        down = gr.get_boundary([gr.faces[4]])
        move_ants(self, [up[1]], almost=True)
        self.wait(2)
        move_ants(self, [up[2]], almost=True, foreground=True)
        move_ants(self, [up[3]], almost=True)
        self.wait(3)
        move_ants(self, up)
        self.wait(2)
        a = 4 * [None]
        a[0] = move_ants(self, [down[1]], almost=True, play=False)
        a[1] = move_ants(self, [down[2]], almost=True, foreground=True, play=False)
        a[2] = move_ants(self, [down[3]], almost=True, play=False)
        a[3] = move_ants(self, down, play=False)
        anim = AnimationGroup(*[AnimationGroup(*a[i]) for i in range(4)], lag_ratio=0)
        self.play(anim)

        # 2-cycle examples
        self.graph.highlight_edges(up + down, animate=False)
        self.wait(4)
        self.graph.erase_edge_highlights()

        arrows = gr.add_arrows(up + down, color=GREEN)
        move_ants(self, up + down)
        self.wait()

        reset_ants(self)
        self.remove(*arrows)
        arrows = gr.add_arrows(inv(up + down), color=RED)
        move_ants(self, inv(up + down))
        self.wait()

        text = TextMobject(r"$ 2^{\#cycles} $")
        text.to_corner(UL, buff=LARGE_BUFF)
        self.add(text)

        reset_ants(self)
        self.remove(*arrows)
        arrows = gr.add_arrows(inv(up), color=RED)
        arrows.extend(gr.add_arrows(down, color=GREEN))
        move_ants(self, inv(up) + down)
        self.wait()

        reset_ants(self)
        self.remove(*arrows)
        arrows = gr.add_arrows(up, color=GREEN)
        arrows.extend(gr.add_arrows(inv(down), color=RED))
        move_ants(self, up + inv(down))
        self.wait(4)
        self.remove(*arrows)
        self.remove(text)

        # enclosed faces logic
        remove_ants(self)
        kwargs = {"animate": False, "reset": False}
        gr.anim_2d_3d(run_time=1)
        self.play(self.camera.phi_tracker.set_value, PI / 2 - 0.6)
        #self.play(self.camera.theta_tracker.set_value, PI / 2 - 1)
        #self.play(self.camera.distance_tracker.set_value, 6)
        self.wait(4)
        for opp in ((2, 4), (0, 3), (1, 5)):
            gr.highlight_faces(itemgetter(*opp)(gr.faces), **kwargs)
            self.wait(4 if opp == (2, 4) else 1)
            if opp == (0, 3):
                self.bring_to_back(*self.ants)
            else:
                self.bring_to_front(*self.ants)
            gr.erase_all()

        gr.highlight_faces(itemgetter(3, 5)(gr.faces), **kwargs)
        gr.circle_ant(0)
        gr.circle_ant(1)
        self.wait(1)
        gr.erase_all()

        gr.highlight_faces(itemgetter(2, 3, 4)(gr.faces), **kwargs)
        self.wait(2)

        # 1-cycle example
        cover = gr.get_boundary(itemgetter(2, 3, 4)(gr.faces))
        gr.highlight_edges(cover, animate=False)
        self.wait(2)
        gr.erase_face_highlights()
        move_ants(self, cover)

        self.wait()
        reset_ants(self)
        move_ants(self, inv(cover))
        self.wait(2)
        gr.erase_all()
        self.wait(2)

        # enumerate hamiltonian cycles
        remove_ants(self)
        for f in range(6):
            gr.highlight_faces([gr.faces[f]], **kwargs)
            self.wait(0.4 if f < 5 else 3)
            gr.erase_all()

        gr.highlight_faces(itemgetter(5, 0, 3)(gr.faces), **kwargs)
        self.wait(2)
        gr.erase_all()

        self.wait(2)
        # gr.highlight_edges(gr.get_boundary(itemgetter(5, 2, 4)(gr.faces)),
        #                    foreground=True, stroke_opacity=0.4, animate=False)
        gr.highlight_faces(itemgetter(5, 2, 4)(gr.faces), **kwargs)
        newhl = [Polygon(*[gr.vertices_3d[i] for i in f],
                         fill_opacity=0.5, shade_in_3d=True, rm_inedges=False, stroke_color="#666666")
                 for f in itemgetter(0, 1, 3)(gr.faces)]
        self.add(*newhl)
        self.play(AnimationGroup(
            *[FadeOut(o, run_time=5) for o in gr.drawings[1][FACE_HIGHLIGHTS]],
            *[FadeIn(o, run_time=5) for o in newhl],
        ))
        self.remove(*newhl)
        gr.erase_face_highlights()
        #gr.highlight_faces(itemgetter(0, 1, 3)(gr.faces), **kwargs)
        gr.erase_edge_highlights(animate=True)
        self.wait()

        self.wait(3)
        self.play(*[FadeOut(mob) for mob in self.mobjects])


class CubeEnumScene(Scene):
    def construct(self):
        covers = MyCubeGraph(self).find_cycle_covers()
        twoc = [c for c in covers if nx.number_connected_components(nx.Graph(c)) == 2]
        ham = [c for c in covers if nx.number_connected_components(nx.Graph(c)) == 1]
        texts, anim = [], []
        for i, c in enumerate(twoc):
            cube = MyCubeGraph(self)
            cube.vertices = [0.25 * v for v in cube.vertices]
            cube.vertices = [np.array([v[0] - 4.5, v[1] - 2.5 + 2 * i, 0]) for v in cube.vertices]
            cube.draw_graph(stroke_width=2)
            anim.append(cube.highlight_edges(list(c), animate=False))
            text = TextMobject("4")
            text.next_to(Group(*cube.drawings[0][GRAPH]), direction=2 * RIGHT)
            texts.append(text)
        self.play(AnimationGroup(*anim, lag_ratio=0.2))
        self.add(*texts)
        self.play(AnimationGroup(*[ShowCreation(t) for t in texts], lag_ratio=0.2))
        self.wait()

        texts, anim = [], []
        for i, c in enumerate(ham):
            cube = MyCubeGraph(self)
            cube.vertices = [0.25 * v for v in cube.vertices]
            cube.vertices = [
                np.array([v[0] - 2.5 + 3 * (i // 3 + 1), v[1] - 2.5 + 2 * (i % 3), 0])
                for v in cube.vertices
            ]
            cube.draw_graph(stroke_width=2)
            anim.append(cube.highlight_edges(list(c), animate=False))
            text = TextMobject("2")
            text.next_to(Group(*cube.drawings[0][GRAPH]), direction=2 * RIGHT)
            texts.append(text)
        self.play(AnimationGroup(*anim, lag_ratio=0.2))
        self.add(*texts)
        self.play(AnimationGroup(*[ShowCreation(t) for t in texts], lag_ratio=0.2))
        self.wait()

        text = TextMobject(r"$p = \frac{3 \cdot 4 + 6 \cdot 2}{3^8} = \frac{24}{6561} \approx 0.37\%$")
        text.to_edge(UP)
        self.add(text)
        self.play(ShowCreation(text))
        self.wait(4)
        self.play(*[FadeOut(mob) for mob in self.mobjects])


class PromoScene(ThreeDScene):
    """ python -m manim marching_ants.py PromoScene -pl """

    def construct(self):
        gr = MyDodecahedronGraph(self, is_3d=True)
        gr.vertices_3d =  [1.2 * v for v in gr.vertices_3d]
        gr.vertices_3d = [rotation_matrix(PI / 6, X_AXIS).dot(v) for v in gr.vertices_3d]
        gr.vertices_3d = [rotation_matrix(1 / 3 * PI, Z_AXIS).dot(v) for v in gr.vertices_3d]
        # gr.highlight_faces(gr.faces, fill_opacity=0.8, shade_in_3d=True,
        #                    rm_inedges=False, stroke_color="#999999", animate=False)
        # gr.anim_rotating(axis=Y_AXIS, run_time=7, radians = -PI / 5)
        # gr.vertices_3d = [rotation_matrix(-PI / 5, Y_AXIS).dot(v) for v in gr.vertices_3d]

        covers = gr.find_cycle_covers()
        tric = [c for c in covers if nx.number_connected_components(nx.Graph(c)) == 3]
        ham = [c for c in covers if nx.number_connected_components(nx.Graph(c)) == 1]

        gr.draw_graph(stroke_width=8, bg={2, 3, 7, 8, 14, 19})
        self.wait()

        ants = [create_ant(v, scale=0.3) for v in gr.vertices_3d]
        self.add(*ants)
        self.play(FadeIn(Group(*ants)))
        self.wait()
        move_ants(self, gr.get_boundary(itemgetter(2, 4, 5, 7, 10, 11)(gr.faces)), ants=ants, graph=gr, foreground=True, play=True, run_time=1.5)
        self.wait()
        self.remove(*ants)

        for cover in tric:
            gr.erase_edge_highlights()
            gr.highlight_edges(list(cover), animate=False, bg={2, 3, 7, 8, 14, 19}, color=YELLOW, stroke_width=8)
            self.wait(1.5)
        for cover in ham:
            gr.erase_edge_highlights()
            gr.highlight_edges(list(cover), animate=False, bg={2, 3, 7, 8, 14, 19}, color=YELLOW, stroke_width=8)
            self.wait()
        self.wait(5)


class DodecahedronScene(ThreeDScene):
    """ python -m manim marching_ants.py DodecahedronScene -pl """

    def construct(self):
        intro(self, r"REGULAR DODECAHEDRON")

        # intro and 3D to 2D transformation
        gr = MyDodecahedronGraph(self, is_3d=True)
        gr.vertices_3d = [rotation_matrix(-PI / 6, X_AXIS).dot(v) for v in gr.vertices_3d]
        gr.vertices_3d = [rotation_matrix(1 / 3 * PI, Z_AXIS).dot(v) for v in gr.vertices_3d]
        gr.highlight_faces(gr.faces, fill_opacity=0.5, shade_in_3d=True,
                           rm_inedges=False, stroke_color="#999999", animate=False)
        gr.anim_rotating(axis=Y_AXIS, run_time=7, radians = -PI / 5)
        gr.vertices_3d = [rotation_matrix(-PI / 5, Y_AXIS).dot(v) for v in gr.vertices_3d]
        self.wait(2)

        gr.draw_graph()
        VGroup(*gr.drawings[1][GRAPH]).set_opacity(0)
        vt = ValueTracker(0.5)
        for h in gr.drawings[1][FACE_HIGHLIGHTS]:
            h.add_updater(lambda m: m.set_opacity(vt.get_value()))
        for h in gr.drawings[1][GRAPH]:
            h.add_updater(lambda m: m.set_opacity(1 - 2 * vt.get_value()))
        self.play(vt.set_value, 0, run_time=2, rate_func=lambda t: rush_into(t, inflection=8))
        self.wait(0.5)

        #gr.highlight_edges(gr.edges[:5], stroke_color=YELLOW, animate=True)
        gr.erase_face_highlights()
        self.wait()
        gr.anim_2d_3d(run_time=3)
        self.wait()
        gr.erase_edge_highlights(animate=True)
        self.wait()

        # enclosed faces possibilities and elimination
        ### 1 / 11
        gr.highlight_faces([gr.faces[0]])
        self.wait()

        gr.highlight_faces([gr.faces[6]])
        self.wait()
        gr.circle_vertices([6])
        self.wait()
        gr.cross_edges([5, 6])
        self.wait()
        gr.erase_all()

        ### 2 / 10
        gr.highlight_faces(itemgetter(0, 5)(gr.faces))
        self.wait()
        gr.cross_edges([6, 8, 11, 14, 16, 24])
        self.wait()
        gr.circle_vertices([5, 8, 9, 11, 14, 15])
        self.wait()
        gr.highlight_edges([(16, 11), (11, 5), (5, 12), (12, 9), (9, 13), (13, 8), (8, 14), (14, 19)])
        self.wait()
        gr.highlight_edges([(19, 15), (15, 16)])
        self.wait()
        gr.circle_vertices([17, 18], color=RED)
        self.wait()
        gr.erase_all()

        ### 3 / 7
        gr.highlight_faces(itemgetter(0, 5, 1)(gr.faces))
        self.wait()
        gr.circle_vertices([1], color=RED)
        self.wait()
        gr.erase_all()

        gr.highlight_faces(itemgetter(0, 2, 5)(gr.faces))
        self.wait()
        gr.circle_vertices([11], color=RED)
        self.wait()
        gr.erase_all()

        ### 4 / 8
        gr.highlight_faces(itemgetter(0, 2, 5, 6)(gr.faces))
        self.wait()
        gr.circle_vertices([15], color=RED)
        self.wait()
        gr.erase_all()

        gr.highlight_faces(itemgetter(0, 2, 5, 1)(gr.faces))
        self.wait()
        gr.circle_vertices([0, 1], color=RED)
        self.wait()
        gr.erase_all()

        ### 5 / 7
        gr.highlight_faces(itemgetter(0, 2, 5, 6, 8)(gr.faces))
        self.wait()
        gr.circle_vertices([8], color=RED)
        self.wait()
        gr.erase_all()

        ### 6 / 6
        ## show pattern and move to 3D
        gr.erase(GRAPH)  # TODO find what got broken
        gr.draw_graph(fill_opacity=0.1)
        gr.highlight_faces(itemgetter(0, 1, 2, 3, 4, 5)(gr.faces))
        self.wait()
        gr.highlight_edges([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (15, 16), (16, 17), (17, 18), (18, 19), (19, 15)])
        self.wait()

        gr.erase_all()
        gr.highlight_faces(itemgetter(0, 1, 3, 6, 8, 9)(gr.faces))
        self.wait(3)
        gr.erase_all(animate=True)
        kwargs_3d = dict(fill_opacity=0.5, shade_in_3d=True, rm_inedges=False, stroke_color="#666666")
        gr.highlight_faces(itemgetter(0, 1, 3, 6, 8, 9)(gr.faces), **kwargs_3d, animate=False)
        self.wait()
        gr.erase(GRAPH)
        gr.anim_2d_3d(run_time=3)
        self.wait()

        ## 120 options
        self.remove(*gr.drawings[0][FACE_HIGHLIGHTS])  # TODO find what got broken
        kwargs_3d["animate"] = False
        gr.highlight_faces([gr.faces[6]], **kwargs_3d)
        self.wait(2)

        for i in (2, 10, 11, 7, 1):
            gr.erase_face_highlights()
            gr.highlight_faces(itemgetter(6, i)(gr.faces), **kwargs_3d)
            self.wait(0.5)
        self.wait(2)
        for i in (5, 0, 5, 0):
            gr.erase_face_highlights()
            gr.highlight_faces(itemgetter(6, 1, i)(gr.faces), **kwargs_3d)
            self.wait(0.5)
        self.wait(2)
        for i in (3, 4, 3, 4):
            gr.erase_face_highlights()
            gr.highlight_faces(itemgetter(6, 1, 0, i)(gr.faces), **kwargs_3d)
            self.wait(0.5)
        self.wait(2)
        for i in (8, 9):
            gr.erase_face_highlights()
            gr.highlight_faces(itemgetter(6, 1, 0, 4, i)(gr.faces), **kwargs_3d)
            self.wait(1)
        gr.erase_face_highlights()
        gr.highlight_faces(itemgetter(6, 1, 0)(gr.faces), **kwargs_3d)
        self.wait(1)
        for i in (3, 9, 8):
            gr.highlight_faces([gr.faces[i]], **kwargs_3d)
            self.wait(1)
        self.wait(4)

        ## show symmetries - rotational symmetry
        gr.erase_face_highlights()
        gr.highlight_faces(itemgetter(6, 1, 0, 3, 9, 8)(gr.faces), **kwargs_3d)
        gr.highlight_faces([gr.faces[6]], **kwargs_3d, fill_color=GREEN)
        gr.highlight_faces([gr.faces[8]], **kwargs_3d, fill_color=RED)
        self.wait(4)

        # rotation 1
        axis = gr.vertices_3d[6] - gr.vertices_3d[13]
        axes_kwargs = {"color": WHITE}
        #axisd = Line(gr.vertices_3d[6], gr.vertices_3d[13], **axes_kwargs)

        gr.anim_rotating(axis=axis, run_time=3, radians = TAU / 3)
        gr.vertices_3d = [rotation_matrix(TAU / 3, axis).dot(v) for v in gr.vertices_3d]
        self.wait()

        # rotation 2
        center1 = sum(gr.vertices_3d[i] for i in gr.faces[2]) / 5
        center2 = sum(gr.vertices_3d[i] for i in gr.faces[8]) / 5
        axis = center1 - center2
        #axisd = Line(center1, center2, **axes_kwargs)

        #gr.highlight_edges(gr.get_boundary([gr.faces[8]]), animate=False)
        gr.anim_rotating(axis=axis, run_time=3, radians = -2/5 * TAU)
        gr.erase_edge_highlights()
        gr.vertices_3d = [rotation_matrix(-2/5 * TAU, axis).dot(v) for v in gr.vertices_3d]
        self.wait(4)

        ## show symmetries in 3D - complement + rotational symmetry
        # complement
        gr.highlight_edges(gr.get_boundary(itemgetter(2, 4, 5, 7, 10, 11)(gr.faces)),
                           foreground=True, stroke_opacity=0.4, animate=False)
        newhl = [Polygon(*[gr.vertices_3d[i] for i in f], **kwargs_3d) for f in itemgetter(2, 4, 5, 7, 10, 11)(gr.faces)]
        self.add(*newhl)
        self.play(AnimationGroup(
            *[FadeOut(o, run_time=5) for o in gr.drawings[1][FACE_HIGHLIGHTS]],
            *[FadeIn(o, run_time=5) for o in newhl],
        ))
        self.remove(*newhl)
        gr.erase_face_highlights()
        gr.highlight_faces(itemgetter(2, 4, 5, 7, 10, 11)(gr.faces), **kwargs_3d)
        gr.erase_edge_highlights(animate=True)
        self.wait()

        # rotation 1
        edge1, edge2 = gr.edges[14], gr.edges[20]
        center1 = (gr.vertices_3d[edge1[0]] + gr.vertices_3d[edge1[1]]) / 2
        center2 = (gr.vertices_3d[edge2[0]] + gr.vertices_3d[edge2[1]]) / 2
        axis = center1 - center2
        #axisd = Line(center1, center2, **axes_kwargs)
        self.wait()

        #gr.highlight_edges([gr.edges[14], gr.edges[20]], animate=False)
        gr.anim_rotating(axis=axis, run_time=3, radians=PI)
        gr.erase_edge_highlights()
        gr.vertices_3d = [rotation_matrix(PI, axis).dot(v) for v in gr.vertices_3d]
        self.wait()

        # rotation 2
        center1 = sum(gr.vertices_3d[i] for i in gr.faces[2]) / 5
        center2 = sum(gr.vertices_3d[i] for i in gr.faces[8]) / 5
        axis = center1 - center2
        #axisd = Line(center1, center2, **axes_kwargs)

        #gr.highlight_edges(gr.get_boundary([gr.faces[2]]), animate=False)
        gr.anim_rotating(axis=axis, run_time=3, radians=2 / 5 * TAU)
        gr.erase_edge_highlights()
        gr.vertices_3d = [rotation_matrix(2 / 5 * TAU, axis).dot(v) for v in gr.vertices_3d]
        self.wait(4)
        gr.erase_all(animate=True)

        # show all instances and final result
        covers = gr.find_cycle_covers()
        tric = [c for c in covers if nx.number_connected_components(nx.Graph(c)) == 3]
        ham = [c for c in covers if nx.number_connected_components(nx.Graph(c)) == 1]
        for i, c in enumerate(tric):
            dodeca = MyDodecahedronGraph(self)
            dodeca.vertices = [v / 5 for v in dodeca.vertices]
            dodeca.vertices = [
                np.array([v[0] - 5.5 + 1.5 * i, v[1] + 2.5, 0])
                for v in dodeca.vertices
            ]
            dodeca.draw_graph(stroke_width=1)
            dodeca.highlight_edges(list(c), animate=False, color=GREEN)
            self.wait(0.5 / 6)
        self.wait()
        for i, c in enumerate(ham):
            dodeca = MyDodecahedronGraph(self)
            dodeca.vertices = [v / 6 for v in dodeca.vertices]
            dodeca.vertices = [np.array([
                v[0] - 5.5 + 1.2 * (i % 10),
                v[1] - 2.5 + 1.2 * (i // 10),
                0])
                for v in dodeca.vertices
            ]
            dodeca.draw_graph(stroke_width=1)
            dodeca.highlight_edges(list(c), animate=False)
            self.wait(2 / 30)
        self.wait()

        text = TextMobject(r"$p = \frac{6 \cdot 2^3 + 30 \cdot 2^1}{3^{20}} = \frac{108}{3486784401} \approx 0.000003\%$")
        text.to_edge(UP, buff=2.4)
        self.add(text)
        self.play(ShowCreation(text))
        self.wait(3)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
