Research-quality experiment files used to generate the results for procedural_chlon2026.

# Results manifest

Top-level folders:
- .cursor_like_tool_cache/
- 2026-01-16_compare_experiment_scripts/
- mechanistic_localization/
- prediction1_stage2b/
- prediction2_probes/
- prediction3_checkpointing/

Top-level files:
- finalv4(1).tex
- results_manifest.md

prediction1_stage2b/
- Paper section: Empirical Results -> Prediction 1: Stage~2B Dominates
- gemma/
- llama/
- qwen_merged/
- stage2b_multimodel_suite.py

prediction2_probes/
- Paper section: Empirical Results -> Prediction 2: Probes Certify Information Presence
- routing_multimodel_suite.py
- stage2b_multimodel_suite_probe.py

prediction3_checkpointing/
- Paper section: Empirical Results -> Prediction 3: Checkpointing Recovers Accuracy
- gemma_checkpoint/
- qwen_checkpoint/
- checkpoint_mitigation_suite.py
- checkpoint_multitask_smoke.py

mechanistic_localization/
- Paper section: Empirical Results -> Mechanistic Localization: Activation Patching
- gemma_causal_head/
- qwen_causal_head/
- gemma_mech/
- llama_mech/
- qwen_mech/
- mech_stage2b_circuit_suite.py
- mechanistic_review_pack_numeric_oomsafe.py
- mechanistic_review_pack_prompt_style.py
