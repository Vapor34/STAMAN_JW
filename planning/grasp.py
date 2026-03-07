"""
Progressive Grasp Executor

Implements a simple incremental finger‑closure strategy that can be used
after a pre‑grasp pose has been determined (for example by
``planning.position_planner.PositionPlanner``).

Usage example::

    executor = ProgressiveGrasp(model, data, hand_control)
    pre_pose = state_struct_from_position_planner
    result = executor.execute(pre_pose)
    if result['success']:
        print("grasp obtained with force", result['total_force'])

The algorithm keeps the synergy variables initially at the values
contained in ``pre_pose`` (typically something like half‑closed) and
then gradually ramps them toward 1.0.  At each simulation step it
updates the Mujoco state, steps the simulation, and inspects the
contacts involving the distal links of the fingers.  When the total
normal force on those bodies exceeds ``target_force`` or a
self‑collision is detected, the execution stops.

The return dictionary contains

* ``success`` – whether the target force was reached
* ``final_state`` – a ``StateStruct`` snapshot of the closing hand
* ``total_force`` – measured contact force at termination
* ``steps`` – number of mujoco steps performed
"""

import mujoco
import numpy as np
from core.hand_state import StateStruct
from core.hand_control import HandControl


class ProgressiveGrasp:
    def __init__(self, model, data, hand_control,hand_body_prefix='lh_'):
        """Prepare executor with model/data and a pre‑constructed HandControl."""
        self.model = model
        self.data = data
        self.hand_control = hand_control
        self.hand_prefix = hand_body_prefix

        # Collect hand geometry and contact points
        self.hand_geom_ids = []
        self.contact_body_ids = []
        excluded_keywords = ['wrist', 'forearm', 'palm']
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and self.hand_prefix in name:
                self.hand_geom_ids.extend(self._get_geoms_id_of_body(i))
                name_lower = name.lower()
                is_excluded = any(k in name_lower for k in excluded_keywords)
                if not is_excluded:
                    self.contact_body_ids.append(i)

        # runtime state used for stepped execution
        self._running = False
        self._current_state = None
        self._target_force = None
        self._max_steps = None
        self._steps = 0
        self._increment = None
        self._closure_steps = None

    def _get_geoms_id_of_body(self, body_id):
        start_geom = self.model.body_geomadr[body_id]
        num_geoms = self.model.body_geomnum[body_id]
        return [start_geom + j for j in range(num_geoms)]
        
        
    def execute(self,
                pre_grasp_state: StateStruct,
                target_force: float = 8.0,
                increment: float = 0.005,
                max_steps: int = 1000,
                closure_time: float | None = None):
        """Run the progressive closure loop.

        Args:
            pre_grasp_state: StateStruct containing the hand pose at the start
            target_force: total contact force (N) that we aim to achieve
            increment: amount to increase each relevant synergy variable per step
            max_steps: hard limit on simulation steps

        Returns:
            dict with keys ``success``, ``final_state``, ``total_force`` and
            ``steps``.
        """
        # copy so we don't modify caller's object
        state = pre_grasp_state.copy()

        # Begin with whatever synergy values are stored in the state
        # (typically ~0.5 from the position planner)
        success = False
        total_force = 0.0

        # if closure_time is given, convert it to a per-step increment
        if closure_time is not None:
            # number of simulation steps corresponding to the duration
            dt = self.model.opt.timestep
            nsteps = max(1, int(round(closure_time / dt)))
            increment_grasp = (1.0 - state.grasp_synergy) / nsteps
            increment_curl = (1.0 - state.curl_synergy) / nsteps
            increment_thumb = (1.0 - state.thumb_flex_synergy) / nsteps
        else:
            increment_grasp = increment_curl = increment_thumb = increment

        for step in range(max_steps):
            # bump synergies (clip at 1.0 which is the practical maximum)
            state.grasp_synergy = min(state.grasp_synergy + increment_grasp, 1.0)
            state.curl_synergy = min(state.curl_synergy + increment_curl, 1.0)
            state.thumb_flex_synergy = min(state.thumb_flex_synergy + increment_thumb, 1.0)
            # other synergies remain as-is

            # write into mujoco
            self.hand_control.set_hand_state(state)
            mujoco.mj_step(self.model, self.data)

            total_force = self._measure_finger_force()

            if total_force >= target_force:
                success = True
                break

            if self._detect_self_collision():
                # back off slightly 
                state.grasp_synergy = max(0.0, state.grasp_synergy - increment)
                state.curl_synergy = max(0.0, state.curl_synergy - increment)
                state.thumb_flex_synergy = max(0.0, state.thumb_flex_synergy - increment)
                self.hand_control.set_hand_state(state)
                mujoco.mj_step(self.model, self.data)

        return {
            'success': success,
            'final_state': state,
            'total_force': total_force,
            'steps': step + 1
        }

    # ------------------------------------------------------------------
    # stepped/time‑aware interface
    # ------------------------------------------------------------------
    def start(self,
              pre_grasp_state: StateStruct,
              target_force: float = 8.0,
              closure_time: float | None = None,
              max_steps: int = 1000):
        """Begin a grasp sequence that can be advanced with ``step()``.

        If ``closure_time`` is provided the required per‑step increments are
        computed so that the hand reaches full closure in approximately that
        many *simulated* seconds.  Otherwise a fixed increment of ``0.005``
        is used (same as the old behaviour).

        After calling ``start`` the caller should repeatedly call ``step()``
        until it returns a non‑``None`` result, indicating completion.
        """
        self._current_state = pre_grasp_state.copy()
        self._target_force = target_force
        self._max_steps = max_steps
        self._steps = 0
        self._running = True

        # compute increments
        if closure_time is not None:
            dt = self.model.opt.timestep
            nsteps = max(1, int(round(closure_time / dt)))
            self._increment = {
                'grasp': (1.0 - self._current_state.grasp_synergy) / nsteps,
                'curl': (1.0 - self._current_state.curl_synergy) / nsteps,
                'thumb_flex': (1.0 - self._current_state.thumb_flex_synergy) / nsteps,
            }
            self._closure_steps = nsteps
        else:
            self._increment = {'grasp': 0.005, 'curl': 0.005, 'thumb_flex': 0.005}
            self._closure_steps = None

        # return nothing; caller drives with step()

    def step(self):
        """Advance the grasp by one simulation step.

        Returns a result dict identical to ``execute`` once the grasp has
        finished (either by reaching ``target_force`` or hitting ``max_steps``
        or detecting self‑collision).  Returns ``None`` while the sequence is
        still in progress.
        """
        if not self._running:
            return None

        self._steps += 1
        inc = self._increment
        self._current_state.grasp_synergy = min(self._current_state.grasp_synergy + inc['grasp'], 1.0)
        self._current_state.curl_synergy = min(self._current_state.curl_synergy + inc['curl'], 1.0)
        self._current_state.thumb_flex_synergy = min(self._current_state.thumb_flex_synergy + inc['thumb_flex'], 1.0)

        # apply to Mujoco and step
        self.hand_control.set_hand_state(self._current_state)
        mujoco.mj_step(self.model, self.data)

        total_force = self._measure_finger_force()
        done = False
        success = False
        if total_force >= self._target_force:
            done = True
            success = True
        elif self._detect_self_collision():
            done = True
            # back off as before
            self._current_state.grasp_synergy = max(0.0, self._current_state.grasp_synergy - inc['grasp'])
            self._current_state.curl_synergy = max(0.0, self._current_state.curl_synergy - inc['curl'])
            self._current_state.thumb_flex_synergy = max(0.0, self._current_state.thumb_flex_synergy - inc['thumb_flex'])
            self.hand_control.set_hand_state(self._current_state)
            mujoco.mj_step(self.model, self.data)
        elif self._steps >= self._max_steps:
            done = True

        if done:
            self._running = False
            return {
                'success': success,
                'final_state': self._current_state.copy(),
                'total_force': total_force,
                'steps': self._steps
            }
        else:
            return None

    def _measure_finger_force(self) -> float:
        """Sum the normal forces"""
        total = 0.0
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            body1 = self.model.geom_bodyid[con.geom1]
            body2 = self.model.geom_bodyid[con.geom2]
            if body1 in self.contact_body_ids or body2 in self.contact_body_ids:
                # ``con.H`` is the contact impulse/force array (normal + tangential)
                total += np.linalg.norm(np.array(con.H))
        return total

    def _detect_self_collision(self) -> bool:
        """Return True if any contact is between two hand bodies."""
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            b1 = self.model.geom_bodyid[con.geom1]
            b2 = self.model.geom_bodyid[con.geom2]
            name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b1) or ""
            name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b2) or ""
            if self.hand_prefix in name1 and self.hand_prefix in name2:
                # simple heuristic: anything beyond fingertip adjacency is bad
                if con.dist < -1e-4:
                    return True
        return False
