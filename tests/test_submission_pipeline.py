from __future__ import annotations

import unittest
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import hydra
import h5py
import numpy as np
import torch

SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
if str(SUBMISSION_ROOT) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_ROOT))

from diffusion.anchors import fit_action_anchors
from diffusion.pipeline import build_stage_commands
from scripts.split_hdf5_by_episode import main as split_hdf5_main


class KMeansNearestAnchorTests(unittest.TestCase):
    def test_all_anchors_are_real_teacher_action_chunks(self) -> None:
        teacher_plan = torch.tensor(
            [
                [10.0, 0.0],
                [12.0, 0.0],
                [-10.0, 0.0],
                [-12.0, 0.0],
                [0.0, 10.0],
                [0.0, 12.0],
            ],
            dtype=torch.float32,
        )
        bundle = fit_action_anchors(
            teacher_plan,
            num_anchors=3,
            plan_horizon=1,
            action_dim=2,
            seed=0,
            max_iter=50,
        )

        teacher_rows = {tuple(row.tolist()) for row in teacher_plan}
        anchor_rows = {tuple(row.tolist()) for row in bundle.anchors}
        self.assertTrue(anchor_rows.issubset(teacher_rows))
        self.assertEqual(bundle.fit_method, "kmeans_nearest_real_sample")
        self.assertEqual(len(bundle.metadata["selected_teacher_plan_indices"]), 3)
        self.assertGreater(bundle.metadata["centroid_to_real_l2_mean"], 0.0)


class SplitProtocolConfigTests(unittest.TestCase):
    def _compose(self, task: str):
        config_dir = SUBMISSION_ROOT / "config" / "diffusion"
        with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            return hydra.compose(config_name="train", overrides=[f"task={task}"])

    def test_all_main_tasks_use_train_and_test_episode_splits(self) -> None:
        for task in ("cube", "pusht", "reacher", "tworoom"):
            with self.subTest(task=task):
                cfg = self._compose(task)
                commands = build_stage_commands(cfg)

                split_command = commands["split_hdf5"]
                train_h5 = str(cfg.task.split_train_h5)
                test_h5 = str(cfg.task.split_test_h5)
                self.assertEqual(
                    split_command[split_command.index("--output-train-h5") + 1],
                    train_h5,
                )
                self.assertEqual(
                    split_command[split_command.index("--output-test-h5") + 1],
                    test_h5,
                )

                dataset_command = commands["build_dataset"]
                self.assertEqual(
                    dataset_command[dataset_command.index("--dataset-h5") + 1],
                    train_h5,
                )
                self.assertIn(f"dataset_h5={test_h5}", commands["eval"])
                self.assertIn("kmeans_nearest", str(cfg.task.anchor_bundle_path))
                anchor_command = commands["build_anchors"]
                self.assertEqual(
                    anchor_command[anchor_command.index("--selection") + 1],
                    "kmeans-nearest",
                )
                self.assertNotIn("split_val_h5", cfg.task)


class EpisodeSplitIntegrationTests(unittest.TestCase):
    def test_split_keeps_complete_episodes_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_h5 = root / "dataset.h5"
            train_h5 = root / "train" / "dataset.h5"
            test_h5 = root / "test" / "dataset.h5"
            episode_ids = np.repeat(np.arange(4, dtype=np.int64), 3)
            actions = np.stack(
                [episode_ids.astype(np.float32), np.arange(12, dtype=np.float32)],
                axis=1,
            )
            with h5py.File(input_h5, "w") as handle:
                handle.create_dataset("episode_idx", data=episode_ids)
                handle.create_dataset("action", data=actions)
                handle.create_dataset("ep_len", data=np.full(4, 3, dtype=np.int64))
                handle.create_dataset("ep_offset", data=np.arange(0, 12, 3, dtype=np.int64))

            argv = [
                "split_hdf5_by_episode.py",
                "--input-h5",
                str(input_h5),
                "--output-train-h5",
                str(train_h5),
                "--output-test-h5",
                str(test_h5),
                "--train-ratio",
                "0.5",
                "--seed",
                "42",
            ]
            with patch.object(sys, "argv", argv):
                split_hdf5_main()

            with h5py.File(train_h5, "r") as handle:
                train_original_episode_ids = set(handle["action"][:, 0].astype(int).tolist())
            with h5py.File(test_h5, "r") as handle:
                test_original_episode_ids = set(handle["action"][:, 0].astype(int).tolist())

            self.assertFalse(train_original_episode_ids & test_original_episode_ids)
            self.assertEqual(train_original_episode_ids | test_original_episode_ids, set(range(4)))


if __name__ == "__main__":
    unittest.main()
