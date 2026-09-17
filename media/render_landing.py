import sys, os, numpy as np
P = os.environ["MPCPROJ"]; OUT = os.environ["OUTDIR"]
sys.path.insert(0, P)
import pyvista as pv
pv.OFF_SCREEN = True
from src.rocket import Rocket
from src.pos_rocket_vis import RocketVis
d = np.load(os.path.join(OUT, "landing_traj.npz")); T, X, U = d["T"], d["X"], d["U"]
rocket = Rocket(Ts=1/20, model_params_filepath=os.path.join(P, "rocket.yaml")); rocket.mass = 1.7
vis = RocketVis(rocket, os.path.join(P, "Cartoon_rocket.obj"))
pl = pv.Plotter(off_screen=True, window_size=(1280, 720))
ground = pv.Plane(center=[0, 0, 0], direction=[0, 0, 1], i_size=30, j_size=30, i_resolution=15, j_resolution=15)
pl.add_mesh(ground, color="#cfe8cf", opacity=0.7, show_edges=True, edge_color="#9dbb9d")
pl.add_mesh(pv.Cylinder(center=(1.0, 0.0, 0.02), direction=(0, 0, 1), radius=0.7, height=0.04), color="dimgray")
pl.add_mesh(pv.Sphere(radius=0.12, center=(1.0, 0.0, 3.0)), color="black")  # target
rocket_actor, mesh_pts = vis._create_rocket_mesh(pl)
xa, ya, za, xp, yp, zp = vis._create_rocket_axis(pl)
traj_actor = vis._create_trajectory(pl)
thrust_actor, thrust_pts = vis._create_thrustvector(pl)
scene = dict(rocket_actor=rocket_actor, rocket_mesh_points=mesh_pts, x_axis_actor=xa, y_axis_actor=ya, z_axis_actor=za,
             x_axis_mesh_points=xp, y_axis_mesh_points=yp, z_axis_mesh_points=zp, trajectory_actor=traj_actor,
             thrust_actor=thrust_actor, thrust_mesh_points=thrust_pts)
traj_actor.prop.line_width = 4; traj_actor.prop.color = "royalblue"
pl.set_background("white"); pl.add_axes(line_width=2)
pl.camera.position = (21.0, -15.0, 11.0); pl.camera.focal_point = (1.5, 1.0, 5.5); pl.camera.up = (0, 0, 1)
txt = pl.add_text("", position="upper_left", font_size=13, color="black")
sub = pl.add_text("Start (3, 2, 10) m, 30 deg roll   ->   target (1, 0, 3) m", position="lower_left", font_size=10, color="dimgray")
fps = 20
pl.open_movie(os.path.join(OUT, "rocket_landing_nmpc.mp4"), framerate=fps, quality=8)
traj = np.column_stack([X[9], X[10], X[11]])
for k in range(X.shape[1]):
    vis._update_rocket_pose(scene, traj[k], X[3:6, k], U[0, k], U[1, k], U[2, k])
    vis._update_trajectory(scene, traj[:k+1])
    txt.SetText(2, f"Nonlinear MPC landing (EPFL ME-425 project)   t = {T[k]:4.1f} s   z = {X[11,k]:4.1f} m")
    pl.write_frame()
    if k in (10, 190): pl.screenshot(os.path.join(OUT, f"check_{k}.png"))
for _ in range(fps): pl.write_frame()
pl.close(); print("movie done")
