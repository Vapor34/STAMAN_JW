"""Grasp quality evaluation based on force-closure analysis.

Pipeline:
    1. Extract contact data (position, normal, force) from MuJoCo.
    2. Build the *wrench matrix*  W ∈ ℝ^{6 × k}  using linearised
       friction cones at each contact point.
    3. Check force closure: the positive span of W must contain the
       origin in ℝ⁶, i.e. any external wrench can be balanced.

Force-closure test uses linear programming (LP):
    max  ε
    s.t. W λ = ε · d,   λ ≥ 0,   Σλ ≤ 1    ∀ unit direction d

Instead of testing all d, we solve 12 LPs (±  along each of the
6 wrench axes).  ε > 0 for every direction ⟹ force closure.
The minimum ε across all 12 directions is the *closure margin* –
a scalar quality score (0 = not closed, higher = better).
"""

import mujoco
import numpy as np
from scipy.optimize import linprog


class GraspEvaluator:
    """Evaluate grasp quality via force-closure analysis.

    Parameters
    ----------
    model : mujoco.MjModel
    data : mujoco.MjData
    object_body_name : str
        Name of the grasped object body.
    hand_body_prefix : str
        Prefix used to identify hand bodies (default ``"lh_"``).
    friction_coeff : float
        Coulomb friction coefficient μ used to linearise the friction cone.
    cone_edges : int
        Number of edges for the polyhedral friction-cone approximation.
    """

    def __init__(
        self,
        model,
        data,
        object_body_name: str,
        hand_body_prefix: str = "lh_",
        friction_coeff: float = 0.5,
        cone_edges: int = 8,
    ):
        self.model = model
        self.data = data
        self.obj_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, object_body_name
        )
        self.hand_body_prefix = hand_body_prefix
        self.friction_coeff = friction_coeff
        self.cone_edges = cone_edges

    # ------------------------------------------------------------------
    # 1. Contact data extraction
    # ------------------------------------------------------------------

    def _get_contacts(self):
        """Return a list of (position, normal, force_magnitude) tuples.

        Iterates over all active contacts.  Only contacts between the
        hand and the target object are kept.  The contact normal is
        oriented *into* the object (pointing from hand → object).

        Returns
        -------
        list[tuple[np.ndarray, np.ndarray, float]]
            Each entry is ``(pos[3], normal[3], force_scalar)``.
        """
        contacts = []
        force_buf = np.zeros(6)

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            body1 = self.model.geom_bodyid[con.geom1]
            body2 = self.model.geom_bodyid[con.geom2]

            if self.obj_body_id not in (body1, body2):
                continue

            # Identify which side is hand, which is object
            hand_body = body1 if body1 != self.obj_body_id else body2
            body_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, hand_body
            )
            if body_name is None:
                continue
            if not body_name.lower().startswith(self.hand_body_prefix):
                continue

            # Contact force magnitude
            mujoco.mj_contactForce(self.model, self.data, i, force_buf)
            force_mag = np.linalg.norm(force_buf[:3])
            if force_mag < 1e-6:
                continue

            # Contact position (world frame)
            pos = con.pos.copy()

            # Contact normal: frame[0:3] is the contact normal.
            # MuJoCo convention: normal points from geom1 → geom2.
            normal = con.frame[:3].copy()
            # We want normal pointing into the object (hand → object).
            if body1 == self.obj_body_id:
                # geom1 is object, normal points object → hand → flip
                normal = -normal

            contacts.append((pos, normal, force_mag))

        return contacts

    # ------------------------------------------------------------------
    # 2. Wrench matrix construction
    # ------------------------------------------------------------------

    def _build_wrench_matrix(self, contacts):
        """Build the wrench matrix W from contacts with friction cones.

        For each contact point the Coulomb friction cone is
        approximated by ``cone_edges`` generators equally spaced
        around the contact normal.  Each generator produces one
        6D wrench column ``[f; r × f]`` (force + torque about the
        object centre of mass).

        Parameters
        ----------
        contacts : list[tuple]
            Output of :meth:`_get_contacts`.

        Returns
        -------
        np.ndarray, shape (6, k)
            Wrench matrix where k = len(contacts) * cone_edges.
        """
        if not contacts:
            return np.zeros((6, 0))

        obj_com = self.data.xipos[self.obj_body_id].copy()
        mu = self.friction_coeff
        m = self.cone_edges
        wrenches = []

        for pos, normal, _ in contacts:
            # Build a local frame at the contact: normal + two tangents
            t1, t2 = self._tangent_frame(normal)

            # Discretise friction cone into m generators
            for j in range(m):
                angle = 2.0 * np.pi * j / m
                # Generator direction: normal + μ * (cosθ t1 + sinθ t2)
                f_dir = normal + mu * (np.cos(angle) * t1 + np.sin(angle) * t2)
                f_dir /= np.linalg.norm(f_dir)  # unit force

                # Torque about object COM
                r = pos - obj_com
                torque = np.cross(r, f_dir)

                wrench = np.concatenate([f_dir, torque])
                wrenches.append(wrench)

        # shape (6, k)
        return np.array(wrenches).T

    @staticmethod
    def _tangent_frame(normal):
        """Return two unit tangent vectors orthogonal to *normal*."""
        n = normal / np.linalg.norm(normal)
        # Pick an arbitrary vector not parallel to n
        if abs(n[0]) < 0.9:
            aux = np.array([1.0, 0.0, 0.0])
        else:
            aux = np.array([0.0, 1.0, 0.0])
        t1 = np.cross(n, aux)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        t2 /= np.linalg.norm(t2)
        return t1, t2

    # ------------------------------------------------------------------
    # 3. Force-closure test (LP-based)
    # ------------------------------------------------------------------

    def _check_force_closure(self, W):
        """Test whether the wrench set W achieves force closure.

        For each of the 12 axis-aligned unit directions d in ℝ⁶ we
        solve:

            max  ε
            s.t. W λ  = ε · d
                 λ  ≥ 0
                 1ᵀλ ≤ 1          (bounded effort)

        Force closure holds iff the optimum ε > 0 for **all** 12
        directions.  The minimum ε is the *closure margin*.

        Parameters
        ----------
        W : np.ndarray, shape (6, k)

        Returns
        -------
        is_closed : bool
        margin : float
            Minimum ε across all directions (≤ 0 if not closed).
        """
        if W.shape[1] == 0:
            return False, 0.0

        n_cols = W.shape[1]
        margins = []

        for axis in range(6):
            for sign in (+1.0, -1.0):
                d = np.zeros(6)
                d[axis] = sign

                # Decision variables: x = [λ (k), ε (1)]
                # Minimise -ε  ⟺  maximise ε
                c = np.zeros(n_cols + 1)
                c[-1] = -1.0  # coefficient for ε

                # Equality: W λ - ε d = 0  →  [W | -d] x = 0
                A_eq = np.hstack([W, -d.reshape(6, 1)])
                b_eq = np.zeros(6)

                # Inequality: 1ᵀλ ≤ 1 (exclude ε from sum)
                A_ub = np.zeros((1, n_cols + 1))
                A_ub[0, :n_cols] = 1.0
                b_ub = np.array([1.0])

                # Bounds: λ ≥ 0, ε free (but practically bounded)
                bounds = [(0.0, None)] * n_cols + [(None, None)]

                res = linprog(
                    c,
                    A_ub=A_ub,
                    b_ub=b_ub,
                    A_eq=A_eq,
                    b_eq=b_eq,
                    bounds=bounds,
                    method="highs",
                )

                if res.success:
                    epsilon = res.x[-1]
                else:
                    epsilon = 0.0

                margins.append(epsilon)

        min_margin = min(margins)
        is_closed = min_margin > 1e-6
        return is_closed, min_margin

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self):
        """Run the full force-closure evaluation pipeline.

        Returns
        -------
        result : dict
            ``"is_force_closure"`` : bool
            ``"closure_margin"``   : float  (quality score, >0 = closed)
            ``"num_contacts"``     : int
            ``"contact_bodies"``   : list[str]  (names of hand bodies in contact)
        """
        contacts = self._get_contacts()

        # Collect contact body names for reporting
        contact_bodies = set()
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            b1 = self.model.geom_bodyid[con.geom1]
            b2 = self.model.geom_bodyid[con.geom2]
            if self.obj_body_id not in (b1, b2):
                continue
            hand_id = b1 if b1 != self.obj_body_id else b2
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, hand_id)
            if name and name.lower().startswith(self.hand_body_prefix):
                contact_bodies.add(name)

        W = self._build_wrench_matrix(contacts)
        is_closed, margin = self._check_force_closure(W)

        return {
            "is_force_closure": is_closed,
            "closure_margin": margin,
            "num_contacts": len(contacts),
            "contact_bodies": sorted(contact_bodies),
        }