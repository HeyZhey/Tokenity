from __future__ import annotations

import copy
import importlib
import unittest
from types import SimpleNamespace

try:
    import mlx.core as mx
except ImportError:  # ordinary control-plane test environment
    mx = None


@unittest.skipUnless(mx is not None, "MLX runtime is available only on the target Apple Silicon environment")
class NativeMTPMLXRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from mlx_lm.models import qwen3_5

        from tokenity.inference.native_mtp import cache, generation, qwen
        from tokenity.inference.native_mtp.runtime import (
            PatchTransaction,
            native_mtp_construction_scope,
            runtime_shape_supported,
        )

        cls.generate = importlib.import_module("mlx_lm.generate")
        cls.cache_patch = cache
        cls.generation_patch = generation
        cls.native_scope = staticmethod(native_mtp_construction_scope)
        cls.qwen_module = qwen3_5
        cls.transaction = PatchTransaction()
        cache.install(cls.transaction)
        qwen.install(cls.transaction)
        cls.telemetry = {"proposed_tokens": 0, "accepted_tokens": 0}
        generation.install(cls.transaction, cls.telemetry)
        supported, detail = runtime_shape_supported()
        if not supported:
            raise AssertionError(detail)

        cls.text_config = {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 32,
            "full_attention_interval": 2,
            "linear_num_value_heads": 4,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "tie_word_embeddings": False,
            "num_experts": 0,
            "mtp_num_hidden_layers": 1,
        }
        mx.random.seed(7)
        with cls.native_scope(True):
            args = qwen3_5.ModelArgs.from_dict(
                {"model_type": "qwen3_5_moe_text", "text_config": cls.text_config}
            )
            cls.model = qwen3_5.Model(args)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.transaction.rollback()

    @staticmethod
    def greedy_sampler(logprobs):
        return mx.argmax(logprobs, axis=-1)

    def setUp(self) -> None:
        sampler = self.greedy_sampler
        sampler.temp = 0.0
        sampler.top_p = 1.0
        sampler.top_k = 0
        sampler.min_p = 0.0
        sampler.min_tokens_to_keep = 1
        sampler.xtc_probability = 0.0

    def _batch(self, *, active: bool, max_tokens: int = 12, state_machine=None):
        self.model.language_model._tokenity_native_mtp_decode_enabled = active
        state_machine = state_machine or self.generate.SequenceStateMachine()
        prompt = self.generate.PromptProcessingBatch(
            model=self.model,
            uids=[1],
            caches=[self.model.make_cache()],
            tokens=[[]],
            samplers=[self.greedy_sampler],
            fallback_sampler=self.greedy_sampler,
            logits_processors=[[]],
            state_machines=[state_machine],
            max_tokens=[max_tokens],
        )
        return prompt.generate([[1, 2, 3]])

    def _run(self, *, active: bool, max_tokens: int = 12, state_machine=None):
        batch = self._batch(active=active, max_tokens=max_tokens, state_machine=state_machine)
        output = []
        while batch.uids:
            batch._tokenity_native_mtp_activation_safe = True
            output.extend(response.token for response in batch.next())
        return output

    def test_qwen_head_forward_and_default_off_construction(self):
        self.assertTrue(self.model.language_model._tokenity_native_mtp_decode_enabled)
        self.assertEqual(len(self.model.language_model.mtp.layers), 1)
        cache = self.model.make_cache()
        logits, hidden = self.model(
            mx.array([[1, 2]], dtype=mx.uint32),
            cache=cache,
            return_hidden=True,
        )
        head_logits = self.model.mtp_forward(
            hidden[:, -1:],
            mx.array([[3]], dtype=mx.uint32),
            self.model.make_mtp_cache(),
        )
        mx.eval(logits, hidden, head_logits)
        self.assertEqual(logits.shape, (1, 2, 32))
        self.assertEqual(hidden.shape, (1, 2, 32))
        self.assertEqual(head_logits.shape, (1, 1, 32))

        with self.native_scope(False):
            args = self.qwen_module.ModelArgs.from_dict(
                {"model_type": "qwen3_5_moe_text", "text_config": self.text_config}
            )
            ordinary = self.qwen_module.Model(args)
        self.assertFalse(hasattr(ordinary.language_model, "mtp"))

    def test_rejected_verify_restores_hybrid_cache_to_confirmed_prefix(self):
        from mlx.utils import tree_flatten

        baseline = self.model.make_cache()
        self.model(mx.array([[1, 2]], dtype=mx.uint32), cache=baseline)
        mx.eval([entry.state for entry in baseline])
        expected = copy.deepcopy(baseline)
        actual = copy.deepcopy(baseline)

        self.model(mx.array([[3]], dtype=mx.uint32), cache=expected)
        self.cache_patch.set_undo_armed(True)
        try:
            self.model(
                mx.array([[3, 4]], dtype=mx.uint32),
                cache=actual,
                return_hidden=True,
                n_confirmed=1,
            )
        finally:
            self.cache_patch.set_undo_armed(False)
        self.assertTrue(self.cache_patch.restore_after_rejection(actual))
        mx.eval([entry.state for entry in expected], [entry.state for entry in actual])

        for expected_cache, actual_cache in zip(expected, actual):
            self.assertEqual(
                getattr(expected_cache, "offset", None),
                getattr(actual_cache, "offset", None),
            )
            expected_arrays = [value for _, value in tree_flatten(expected_cache.state) if value is not None]
            actual_arrays = [value for _, value in tree_flatten(actual_cache.state) if value is not None]
            self.assertEqual(len(expected_arrays), len(actual_arrays))
            for left, right in zip(expected_arrays, actual_arrays):
                self.assertLess(float(mx.max(mx.abs(left - right)).item()), 1e-5)

    def test_greedy_output_matches_standard_path_and_honors_max_tokens(self):
        standard = self._run(active=False, max_tokens=12)
        native_mtp = self._run(active=True, max_tokens=12)
        self.assertEqual(native_mtp, standard)
        self.assertEqual(len(native_mtp), 12)
        self.assertGreater(int(self.telemetry["proposed_tokens"]), 0)

    def test_stop_sequence_and_cancellation_do_not_over_emit(self):
        baseline = self._run(active=False, max_tokens=10)
        stop_token = baseline[2]
        standard_machine = self.generate.SequenceStateMachine(
            {"normal": [([stop_token], None)]}
        )
        mtp_machine = self.generate.SequenceStateMachine(
            {"normal": [([stop_token], None)]}
        )
        standard = self._run(active=False, max_tokens=10, state_machine=standard_machine)
        native_mtp = self._run(active=True, max_tokens=10, state_machine=mtp_machine)
        self.assertEqual(native_mtp, standard)
        self.assertEqual(native_mtp[-1], stop_token)

        batch = self._batch(active=True, max_tokens=10)
        batch._tokenity_native_mtp_activation_safe = True
        batch.next()
        batch._tokenity_native_mtp_activation_safe = True
        batch.next()
        batch.filter([])
        self.assertEqual(batch.uids, [])
        self.assertFalse(hasattr(batch, "_tokenity_native_mtp_state"))

    def test_lazy_activation_and_late_join_guard(self):
        batch = self._batch(active=True, max_tokens=5)
        batch._tokenity_native_mtp_activation_safe = False
        batch.next()
        self.assertFalse(hasattr(batch, "_tokenity_native_mtp_state"))

        active = self._batch(active=True, max_tokens=5)
        active._tokenity_native_mtp_activation_safe = True
        active.next()
        donor = self._batch(active=True, max_tokens=5)
        with self.assertRaisesRegex(RuntimeError, "late join"):
            active.extend(donor)

    def test_loaded_instance_guard_rejects_random_unloaded_head(self):
        from mlx.utils import tree_flatten

        from tokenity.inference.native_mtp.runtime import (
            NativeMTPRuntimeController,
            NativeMTPStartupError,
        )

        controller = object.__new__(NativeMTPRuntimeController)
        controller.decision = SimpleNamespace(enabled=True)
        with self.assertRaisesRegex(NativeMTPStartupError, "loaded_instance_invalid"):
            controller.validate_loaded_model(self.model)

        self.model._tokenity_native_mtp_loaded_keys = tuple(
            key
            for key, _ in tree_flatten(self.model.parameters())
            if ".mtp." in f".{key}."
        )
        controller.validate_loaded_model(self.model)

    def test_stochastic_acceptance_and_residual_preserve_target_marginal(self):
        target = mx.array([0.6, 0.3, 0.1])
        draft = mx.array([0.2, 0.5, 0.3])
        target_lp = mx.log(target)
        draft_lp = mx.log(draft)

        probabilities = [
            self.generation_patch.acceptance_probability(target_lp, draft_lp, token)
            for token in range(3)
        ]
        residual = self.generation_patch.residual_probabilities(target_lp, draft_lp)
        mx.eval(residual)
        accepted_mass = mx.minimum(target, draft)
        rejected_mass = 1.0 - accepted_mass.sum()
        marginal = accepted_mass + rejected_mass * residual
        self.assertLess(float(mx.max(mx.abs(marginal - target)).item()), 1e-6)
        self.assertEqual(probabilities[0], 1.0)
        self.assertAlmostEqual(probabilities[1], 0.6, places=6)
        self.assertAlmostEqual(probabilities[2], 1.0 / 3.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
