import unittest

from kg.triple_validator import TripleValidator


class OntologyConstraintTests(unittest.TestCase):
    def setUp(self):
        self.validator = TripleValidator()

    def test_rejects_relationships_outside_declared_domain_ranges(self):
        triple = {
            "subject": "recovery rate",
            "subject_type": "Product",
            "relation": "uses",
            "object": "roughing",
            "object_type": "ProcessStage",
        }

        valid, rejected = self.validator.validate([triple])

        self.assertEqual(valid, [])
        self.assertIn("not allowed", rejected[0][1])

    def test_rejects_conflicting_relationships_for_same_entity_pair(self):
        first = {
            "subject": "aeration rate", "subject_type": "Parameter", "relation": "increases",
            "object": "froth depth", "object_type": "Parameter",
        }
        second = dict(first, relation="decreases")

        valid, rejected = self.validator.validate([first, second], check_conflicts=True)

        self.assertEqual(len(valid), 1)
        self.assertEqual(rejected[0][1], "contradicts an existing triple")


if __name__ == "__main__":
    unittest.main()
