"""Continuous retargeting on the frozen pose-origin reaching controller."""
import numpy as np

from reaching_pose_origin import PoseOriginController


def segment(state, goal, duration):
    """Quintic joining arbitrary position/velocity/acceleration to a resting goal."""
    state, goal = np.asarray(state, dtype=float), np.asarray(goal, dtype=float)
    if state.shape != (3, 3) or goal.shape != (3,) or not np.all(np.isfinite(state)) or not np.all(np.isfinite(goal)):
        raise ValueError('state must contain three finite 3-vectors; goal must be finite')
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError('duration must be positive and finite')
    p, v, a = state
    c = np.zeros((6, 3))
    c[:3] = p, v*duration, a*duration**2/2
    c[3:] = np.linalg.solve([[1, 1, 1], [3, 4, 5], [6, 12, 20]],
                            [goal-c[:3].sum(axis=0), -c[1]-2*c[2], -2*c[2]])
    return c


def sample(c, elapsed, duration):
    if not np.isfinite(elapsed):
        raise ValueError('elapsed time must be finite')
    u = np.clip(elapsed/duration, 0, 1)
    p = np.array([1, u, u**2, u**3, u**4, u**5])@c
    v = np.array([0, 1, 2*u, 3*u**2, 4*u**3, 5*u**4])@c/duration
    a = np.array([0, 0, 2, 6*u, 12*u**2, 20*u**3])@c/duration**2
    if elapsed >= duration:
        v, a = np.zeros(3), np.zeros(3)
    return np.array([p, v, a])


EVENTS = [
    dict(name='A', time=0.5, duration=2.0, offset=[.08, 0, 0], hold=[3.5, 4.0]),
    dict(name='B', time=4.0, duration=2.0, offset=[.08, -.04, .02], hold=[6.5, 7.0]),
    dict(name='C', time=7.0, duration=2.0, offset=[.10, -.02, .04], hold=[9.5, 10.0]),
    dict(name='A_cancelled', time=10.0, duration=2.0, offset=[.08, 0, 0], hold=None),
    dict(name='B_retarget', time=10.8, duration=2.0, offset=[.08, -.04, .02], hold=[15.5, 16.0]),
]


class ReactiveController(PoseOriginController):
    def __init__(self, model, config, settings, events=EVENTS):
        super().__init__(model, config, settings)
        self.goal = self.start.copy()
        self.events, self.event_index = events, -1
        self.state = np.array([self.start, np.zeros(3), np.zeros(3)])
        self.coefficients = segment(self.state, self.start, 1)
        self.segment_start, self.segment_duration = 0, 1
        self.goals, self.velocities, self.accelerations, self.event_ids = [], [], [], []

    def _update_reach_reference(self):
        time = self.model.data.time
        # Evaluate the old segment exactly at the event, not at the preceding control tick.
        while self.event_index+1 < len(self.events) and time+1e-10 >= self.events[self.event_index+1]['time']:
            event = self.events[self.event_index+1]
            state = sample(self.coefficients, event['time']-self.segment_start, self.segment_duration)
            self.goal = self.start+np.array(event['offset'])
            self.coefficients = segment(state, self.goal, event['duration'])
            self.segment_start, self.segment_duration = event['time'], event['duration']
            self.event_index += 1
        self.state = sample(self.coefficients, time-self.segment_start, self.segment_duration)
        task = self.reach_task
        task.position_world, task.velocity_world, task.acceleration_world = self.state.copy()
        self.points.append(self.model.attached_point_kinematics(task.body_name, task.point_local)[0].copy())
        self.references.append(task.position_world.copy())
        self.weights.append(task.weight)
        self.guard_risks.append(False)
        self.joint_references.append(self.q_des.copy())
        self.goals.append(self.goal.copy())
        self.velocities.append(task.velocity_world.copy())
        self.accelerations.append(task.acceleration_world.copy())
        self.event_ids.append(self.event_index)
