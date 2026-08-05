# Agent Memory Index

Sessions and their scope. One session file per working session; durable,
verified information is promoted into `HANDOFF.md` and `GENERAL_PROGRESS.md`.

| Session file | Date (UTC) | Agent | Scope / status |
|---|---|---|---|
| `agent_20260804T143457Z_deepseek-v4-flash_3339.md` | 2026-08-04 | deepseek-v4-flash | Initial build of Self Improving Rat (base spec + artificial-life extension). Done — superseded by the 2026-08-05 verification pass. |
| `agent_20260805T101000Z_deepseek-v4-flash_final_audit.md` | 2026-08-05 | deepseek-v4-flash | Parallel verification pass: learning fixes (utility indexing, pred-loss weight, reward_ext, null-guard), scaffolding (CMake/run.sh/main/default.cfg), metrics fixes. Done. |
| `agent_20260805T130000Z_deepseek-v4-flash_verify.md` | 2026-08-05 | deepseek-v4-flash | This session: verification pass — fixed checkpoint serialization (magic skip, utility count) and RingBuffer bugs; wrote the 47-test suite; README; full build/launch/persistence/long-run/no-cheating verification. Done. |

Final repo state: commit `370a58f` (first commit). 47/47 tests pass under
Release and ASan+UBSan. See `GENERAL_PROGRESS.md` for the verified record.
