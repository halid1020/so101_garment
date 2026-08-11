"""Unit tests for the EE-space dataset features (common/recording/features)."""

import unittest

import numpy as np

from common.recording.features import (
    ACTION_EE_KEY,
    EE_DOF,
    EE_NAMES,
    OBS_EE_KEY,
    STATE_NAMES,
    assemble_frame,
    build_dataset_features,
    pose_to_vec7,
)


class TestPoseToVec7(unittest.TestCase):
    def test_identity(self):
        v = pose_to_vec7(np.eye(4))
        np.testing.assert_allclose(v, [0, 0, 0, 1, 0, 0, 0], atol=1e-7)
        self.assertEqual(v.dtype, np.float32)

    def test_translation_and_rotation(self):
        r = np.deg2rad(90.0)
        pose = np.eye(4)
        pose[:3, :3] = [
            [np.cos(r), -np.sin(r), 0],
            [np.sin(r), np.cos(r), 0],
            [0, 0, 1],
        ]
        pose[:3, 3] = [1.0, 2.0, 3.0]
        v = pose_to_vec7(pose)
        np.testing.assert_allclose(v[:3], [1.0, 2.0, 3.0], atol=1e-6)
        # 90° about z → quat (w, x, y, z) = (cos45, 0, 0, sin45).
        np.testing.assert_allclose(
            v[3:], [np.cos(r / 2), 0, 0, np.sin(r / 2)], atol=1e-6
        )

    def test_length_matches_ee_dof(self):
        self.assertEqual(EE_DOF, 14)
        self.assertEqual(len(EE_NAMES), 14)
        self.assertEqual(EE_NAMES[0], "left_x")
        self.assertEqual(EE_NAMES[7], "right_x")


class TestBuildFeaturesEe(unittest.TestCase):
    def test_ee_off_by_default(self):
        feats = build_dataset_features([("scene", 480, 640)])
        self.assertNotIn(OBS_EE_KEY, feats)
        self.assertNotIn(ACTION_EE_KEY, feats)

    def test_ee_features_added(self):
        feats = build_dataset_features([("scene", 480, 640)], include_ee=True)
        for key in (OBS_EE_KEY, ACTION_EE_KEY):
            self.assertIn(key, feats)
            self.assertEqual(feats[key]["shape"], (EE_DOF,))
            self.assertEqual(feats[key]["dtype"], "float32")

    def test_ee_keys_are_neutral_not_action_or_observation(self):
        # Must NOT be picked up by LeRobot's classifier for the joint policy.
        self.assertFalse(OBS_EE_KEY.startswith("observation"))
        self.assertFalse(ACTION_EE_KEY.startswith("action"))


class TestAssembleFrameEe(unittest.TestCase):
    def _state(self):
        return np.zeros(len(STATE_NAMES), dtype=np.float32)

    def test_ee_omitted_by_default(self):
        frame = assemble_frame(self._state(), self._state(), {}, "t")
        self.assertNotIn(OBS_EE_KEY, frame)
        self.assertNotIn(ACTION_EE_KEY, frame)

    def test_ee_included_when_given(self):
        ee = np.arange(EE_DOF, dtype=np.float32)
        tgt = ee + 1
        frame = assemble_frame(
            self._state(), self._state(), {}, "t", ee_pose=ee, ee_target=tgt
        )
        np.testing.assert_allclose(frame[OBS_EE_KEY], ee)
        np.testing.assert_allclose(frame[ACTION_EE_KEY], tgt)
        self.assertEqual(frame[OBS_EE_KEY].dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
