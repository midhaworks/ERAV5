import unittest
from repository_efficiency import common_prefix,parse_calls,execute_calls,prefix_plan
from repository_environment import Environment,initial_messages
from repository_suite import tasks


class EfficiencyChecks(unittest.TestCase):
    def test_prefix_boundaries(self):
        self.assertEqual(common_prefix([1,2,3],[1,2,4]),2)
        self.assertEqual(common_prefix([],[1]),0)
        self.assertEqual(common_prefix([1],[1,2]),1)

    def test_prefix_rewrite_and_shortening_force_reset(self):
        self.assertEqual(prefix_plan([1,2,3,9],[1,2,3],[1,2],True),([1,2,3],2,False))
        self.assertEqual(prefix_plan([1,7,3],[1,7],[1,2],True),([1,7],0,True))
        self.assertEqual(prefix_plan([1,2],[1],[1,2],True),([1],0,True))

    def test_token_boundary_and_nonempty_generation_input(self):
        self.assertEqual(prefix_plan([1,2,9],[1,2,3],[1],True),([1,2],1,False))
        self.assertEqual(prefix_plan([1],[1],[],False),([],0,True))
        with self.assertRaises(ValueError): prefix_plan([],[],[],False)

    def test_read_bundle_matches_single_reads(self):
        with Environment(tasks()[0]) as env:
            calls=[dict(name="read_file",arguments=dict(path=p)) for p in ("api.py","weights.py")]
            before=env.snapshot()
            self.assertEqual(execute_calls(env,calls),[env.execute(c["name"],c["arguments"]) for c in calls])
            self.assertEqual(before,env.snapshot())

    def test_invalid_bundle_cannot_edit_or_finish(self):
        with Environment(tasks()[0]) as env:
            before=env.snapshot()
            for call in [dict(name="edit_file",arguments=dict(path="api.py",code="")),
                         dict(name="finish",arguments={}),dict(name="read_file",arguments=dict(path="../hidden"))]:
                with self.assertRaises(ValueError):
                    execute_calls(env,[dict(name="read_file",arguments=dict(path="api.py")),call])
            self.assertEqual(before,env.snapshot())
            self.assertFalse(env.finished)

    def test_native_framing(self):
        def parser(text,tools): return dict(name="read_file",arguments=dict(path="api.py"))
        block="<tool_call><function=read_file></function></tool_call>"
        self.assertEqual(len(parse_calls(block+"\n"+block,parser)),2)
        for raw in (block+"suffix",block+"stray"+block,block*9):
            with self.assertRaises(ValueError): parse_calls(raw,parser)

    def test_bundled_profile_is_explicit_not_default(self):
        task=tasks()[0]
        self.assertIn("exactly one native tool call",initial_messages(task)[0]["content"])
        self.assertIn("eight read_file calls",initial_messages(task,True)[0]["content"])


if __name__=="__main__": unittest.main()
