import unittest
from repository_comparison import metrics


class ComparisonChecks(unittest.TestCase):
    def test_counts_turns_separately_from_calls(self):
        # Synthetic unit fixture only; never written as execution evidence.
        result=dict(episodes=[dict(passed=True,elapsed_seconds=8,peak_mlx_bytes=1024**3,
            actions=[dict(calls=[dict(name="read_file"),dict(name="read_file")],
                          elapsed_seconds=2,generated_tokens=8),
                     dict(call=dict(name="finish"),elapsed_seconds=1,generated_tokens=2),
                     dict(elapsed_seconds=1,generated_tokens=2)])])
        value=metrics(result)
        self.assertEqual(value["turns"],3)
        self.assertEqual(value["accepted_tool_calls"],3)
        self.assertEqual(value["bundled_read_turns"],1)
        self.assertEqual(value["protocol_invalid_turns"],1)
        self.assertEqual(value["tokens_per_second"],3)
        self.assertEqual(value["generation_seconds"],4)
        self.assertEqual(value["episode_seconds"],8)


if __name__=="__main__": unittest.main()
