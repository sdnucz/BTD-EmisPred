# Matched LightGBM reference results

These files are numerical sources for the existing LightGBM figures. They are
not training targets for reproducing predictions. Targets are taken from the
supplied experimental dataset and fixed partitions.

- `Procedure_Outer_OOF.csv`: one held-out outer prediction for every training record.
- `Independent_Test.csv`: final-model predictions on the independent test records.
- `Eight_Algorithm_Outer_OOF.csv`: separate outer predictions for each candidate.
- `Final_Inner_OOF.csv`: three-fold predictions for each candidate at its saved parameters during final selection.
- `Training_Fit.csv`: final model's fitted training values.
- `External13.csv`, `SI6_Predictions.csv`: structures, conditions, targets and predictions for external examples.
- `Estimator_Parameters.json`: candidate parameters and inner scores for every context.
- `Historical_Config.json`, `Optuna_Trials.csv`: the historical search configuration and trial records.
- `Outer_Algorithm_Selections.csv`: selected candidate in each of ten outer rounds.
- `Selectors.json`: feature-order signatures and scaler values for all 44 selectors.
- `figure_data/`: saved SHAP values, feature values and related figure tables.
- `Provenance.json`: dataset/model identity and the difference between historical fitting and a new eight-algorithm search.
- `SHA256SUMS.json`: hashes checked by `verify_reference.py`.

`record_id` is the cleaned dataset's zero-based row identifier. `outer_fold` and
`inner_fold` in prediction CSVs are one-based. Parameters use `outer/00` through
`outer/09` for those same ten rounds and `final` for full-training selection.

Outer OOF evaluates the complete selection procedure. The eight candidate
comparison is a separate diagnostic; its outer ranking does not select the final
algorithm. Use mean inner-fold RMSE for that choice. Pooled OOF metrics and
unweighted means of fold metrics are distinct summaries.
