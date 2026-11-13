# G3 evaluation

From the repository root, run `evaluate_maps.cmd` in PowerShell as `./evaluate_maps.cmd`.
The defaults are the latest frozen dataset, train split, advanced2, three rounds,
one simulation at a time, one logical CPU and a 4,000,000,000-byte Windows job cap.
Repeat the same command to resume. Completed failures are retained;
`--retry-failed` retries them and `--new` creates a separate batch.

Results, score components, per-drone telemetry, native images, bomb layout figures
and fully populated replay commands are in the batch's `report.md` and JSON files.
The score uses the existing user blue controller and is not a platform score.
`--smoke` is an eight-step diagnostic with a synthetic blue fixture; its full score
is always null. Never compare smoke scores with full rounds.

See the repository-root document `进阶6终极方案_G3_批量对打评测工具与验收.md`
for complete commands, evidence and limitations. Formal batches are user-started.
