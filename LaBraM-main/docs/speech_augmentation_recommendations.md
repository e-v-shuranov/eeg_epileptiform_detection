# Imagined Speech Augmentation Recommendations

The imagined speech dataset shows rapid overfitting once the model starts
adapting to the train split. The lightweight augmentations that were recently
wired into the loader can be enabled with the command-line flags exposed in
`run_class_finetuning_multidata.py`:

```bash
python run_class_finetuning_multidata.py \
  --datasets_dir <PATH_TO_SPEECH_LMDB> \
  --mixup 0.2 \
  --cutmix 0.0 \
  --mixup_prob 0.8 \
  --mixup_switch_prob 0.0 \
  --speech_jitter_std 0.15 \
  --speech_jitter_prob 0.6 \
  --speech_time_shift_pct 0.12 \
  --speech_time_shift_prob 0.7 \
  --speech_channel_dropout_prob 0.3 \
  --speech_channel_dropout_max_pct 0.25
```

## Rationale for the suggested values

* **Mixup 0.2, mixup probability 0.8.** A small amount of mixup encourages
  smoother decision boundaries while keeping the label semantics clear for
  five-class classification. CutMix is left at 0.0 because imagined speech
  segments do not have spatial structure that benefits from rectangular
  swaps, but the switch probability can be revisited if CutMix is enabled.
* **Jitter standard deviation 0.15 with probability 0.6.** Jitter perturbs each
  channel proportionally to its own standard deviation, so a relative
  standard deviation of 0.15 injects noticeable but not destructive noise.
  The 60% probability keeps some clean windows in every batch for stability.
* **Time shift up to 12% with probability 0.7.** Imagined speech commands have
  slight timing variability across trials; rolling by up to ±12% of the
  window length preserves the overall envelope while encouraging temporal
  invariance. The higher probability makes sure many minibatch items receive
  temporal augmentation.
* **Channel dropout probability 0.3 with up to 25% of channels dropped.** EEG
  montages often miss electrodes or suffer from localized artefacts. Randomly
  zeroing up to a quarter of the channels emulates such dropouts and has been
  effective in prior imagined speech work. The 30% trigger rate avoids
  overwhelming the model with heavily corrupted samples.

Feel free to start with these values and adjust them after inspecting training
curves: if the model still overfits quickly, raise the mixup or jitter
probabilities; if optimization becomes unstable, reduce the corresponding
strengths.
