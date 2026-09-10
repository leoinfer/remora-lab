import unittest

from remora.evolution import Candidate, ResurrectionQueue
from remora.manual import AssemblyLedger, ModuleRecord


class LedgerTests(unittest.TestCase):
    def test_transitive_dependents_are_graph_derived(self):
        ledger = AssemblyLedger()
        for name in ("a", "b", "c"):
            ledger.register_module(ModuleRecord(name, "v1", [], [], name, 1))
        ledger.add_dependency("a", "b")
        ledger.add_dependency("b", "c")
        self.assertEqual(ledger.dependents("a"), ["b", "c"])
        self.assertEqual(ledger.minimum_affected_neighborhood("b"), ["b", "c"])

    def test_context_change_increases_resurrection_priority(self):
        queue = ResurrectionQueue()
        queue.add(Candidate("old", "DORMANT", 1.0, 1.0, 1.0, 1.0, {"width": 1}, "narrow"))
        before = queue.priorities({"width": 1})[0]["priority"]
        after = queue.priorities({"width": 4})[0]["priority"]
        self.assertGreater(after, before)


if __name__ == "__main__":
    unittest.main()
