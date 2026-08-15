# Unitree G1 model provenance

The primary model is the official Unitree Robotics `g1_29dof.xml` torque-actuated
MJCF from `unitree_mujoco`, pinned to commit
`ae6a8403e272733e9996ef59990880330496177f`:

<https://github.com/unitreerobotics/unitree_mujoco/tree/ae6a8403e272733e9996ef59990880330496177f/unitree_robots/g1>

The repository supplies the upstream XML, meshes, motor-order document, and
BSD-3-Clause license in this directory. `scene_push_recovery.xml` is the
project-owned scene wrapper: it includes the upstream robot unchanged and adds
only the named ground plane, camera, and neutral lighting needed for this
fixed-foot experiment.

The upstream repository also exposes a file named `g1_23dof.xml`, but that
variant contains six simulator placeholder joints/actuators outside the
physical 23-DoF tree in the current pinned revision. Those placeholders make a
direct floating-base WBC mapping ambiguous. We therefore use the closest
maintained no-hands G1 variant with a physically connected 29-DoF actuator map;
the choice is explicit and the 29-DoF dimensions are tested at load time.

The model is used under the upstream BSD-3-Clause terms. It remains a
simulation model; this project does not claim hardware validation.
