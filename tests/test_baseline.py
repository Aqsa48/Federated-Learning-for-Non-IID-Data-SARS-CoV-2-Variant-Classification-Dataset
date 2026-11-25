import unittest
import numpy as np
from baseline import demo_clients, run, validate


class BaselineTests(unittest.TestCase):
    def test_demo_reproducibility_and_quality(self):
        report = run(demo_clients())
        self.assertEqual(report, run(demo_clients()))
        self.assertGreater(report["macro_f1"], .9)
        self.assertEqual(report["train_rows"] + report["test_rows"], 480)
        self.assertNotIn("SequenceID", report["features"])

    def test_schema_and_labels(self):
        frame = next(iter(demo_clients().values()))
        for column, value in [("Variant", "unknown"), ("feature_0", np.inf)]:
            bad = frame.copy()
            bad.loc[0, column] = value
            with self.assertRaises(ValueError):
                validate(bad)
        bad = frame.copy()
        bad.loc[1, "SequenceID"] = bad.loc[0, "SequenceID"]
        with self.assertRaises(ValueError):
            validate(bad)

    def test_schema_mismatch_and_validation_only(self):
        clients = demo_clients()
        self.assertNotIn("accuracy", run(clients, validate_only=True))
        clients["synthetic-client-1"] = clients["synthetic-client-1"].drop(columns="feature_0")
        with self.assertRaises(ValueError):
            run(clients)


if __name__ == "__main__":
    unittest.main()
