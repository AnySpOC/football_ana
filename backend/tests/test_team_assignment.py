import unittest

from backend.app.services.yolo_overlay import TeamAssignmentSmoother, TeamVisual


class TeamAssignmentSmootherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.team_a = TeamVisual("team_a", (0, 0, 255), (0.0, 0.0, 200.0))
        self.team_b = TeamVisual("team_b", (255, 0, 0), (200.0, 0.0, 0.0))
        self.unknown = TeamVisual("unknown", (160, 160, 160), (0.0, 0.0, 0.0))
        self.smoother = TeamAssignmentSmoother(unknown=self.unknown, switch_frames=3)

    def test_unknown_observation_does_not_replace_known_team(self) -> None:
        self.assertEqual(self.smoother.resolve(7, self.team_a).label, "team_a")
        self.assertEqual(self.smoother.resolve(7, self.unknown).label, "team_a")

    def test_team_changes_only_after_sustained_disagreement(self) -> None:
        self.smoother.resolve(7, self.team_a)

        self.assertEqual(self.smoother.resolve(7, self.team_b).label, "team_a")
        self.assertEqual(self.smoother.resolve(7, self.team_b).label, "team_a")
        self.assertEqual(self.smoother.resolve(7, self.team_b).label, "team_b")

    def test_known_observation_replaces_initial_unknown(self) -> None:
        self.assertEqual(self.smoother.resolve(11, self.unknown).label, "unknown")
        self.assertEqual(self.smoother.resolve(11, self.team_b).label, "team_b")


if __name__ == "__main__":
    unittest.main()
