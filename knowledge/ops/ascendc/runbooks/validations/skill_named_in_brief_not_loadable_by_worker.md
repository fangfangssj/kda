---
type: Runbook
title: "A skill named in a brief is not necessarily loadable by the worker — availability varies per config dir and is invisible from inside the agent; carry a delivery mode"
description: "A skill named in a brief is not necessarily loadable by the worker — availability varies per config dir and is invisible from inside the agent; carry a delivery mode"
tags:
- task.design
- topic.architecture
status: draft
sources: []
created_at: "2026-09-16T00:00:00Z"
updated_at: "2026-10-15T00:00:00Z"
platforms: ["950"]
aliases: ["OL-333"]
---

> **来源说明（2026-10-15）**：本卡内容来自车队私有实测/推理，暂无可公开复核的原始出处；按社区评审要求降为 draft，待补可复核来源后回升。
# OL-333 — OL-333: A5Ops additive extension

## 适用场景

- 适用平台与范围：soc=all。
- 验证来源：卡内无 verified_on 行；有 confirmed_on 行（2026-08-01 (host survey, 4 config dirs)），详见正文。

**OL-333** — **一个 brief 里写出的 skill 名，不等于 worker 能 load 到它**：实测同一台机器四个 config dir，`ascendc-precision-debug` 四个全无，`ascendc-regbase-best-practice` 只有一个有。⇒ MANDATORY invoke 指向一个装不到的 skill，会因为「当初怎么装的」而让本来能跑通的 run 失败，与算子无关。⇒ 而 agent 侧感受不到「我忘了查」——它列表里根本没有，**「没想到查」与「不可能查到」从内部完全同形，只有前者是纪律问题**。做法：投递带 delivery 模式（可 load→invoke；不可 load→给 vendored SKILL.md 绝对路径让它读），读失败只记录不中断。自查：`ls ~/.claude*/skills/

- Former A5Ops ID: `OL-333`
- Fusion disposition: `additive_new`

`applies_to: soc=all` · `confirmed_on: 2026-08-01 (host survey, 4 config dirs)`

**Claim.** Telling a worker to invoke a community Skill is only actionable where that
deployment installed it. Availability varies per config dir, is invisible from inside the
agent, and a mandatory-invoke route pointed at an absent skill fails the run for a reason
that has nothing to do with the operator.

**Measured** (2026-08-01, one host, four config dirs):

```
~/.claude/skills              63 entries   ascendc-regbase-best-practice: no    ascendc-precision-debug: no
~/.claude-account-01/skills   55 entries   ascendc-regbase-best-practice: YES   ascendc-precision-debug: no
~/.claude-ds-op/skills        58 entries   ascendc-regbase-best-practice: no    ascendc-precision-debug: no
~/.claude-kimi/skills         38 entries   ascendc-regbase-best-practice: no    ascendc-precision-debug: no
```

`ascendc-precision-debug` is absent from **all four**. `ascendc-regbase-best-practice` — the
skill carrying RegBase API constraints, a symptom→cause table and DAV_3510 success/failure
comparisons — exists in exactly one.

**Why it matters beyond routing.** An agent that cannot load a skill does not experience
"I forgot to check it". It experiences nothing at all: the skill is not in its list, so no
amount of diligence surfaces it. "Did not think to look" and "could not have looked" are
indistinguishable from the inside, and only the first is a discipline problem. On this host
that distinction covered hours of hand-rediscovering material that was already written down.

**Do.**
- Before relying on a Skill being invocable, check it:
  `ls ~/.claude*/skills/ | grep -E 'regbase|precision-debug|crash-debug|tiling-solver'`
- When surfacing community knowledge to a worker, carry a **delivery mode**. Loadable →
  invoke. Not loadable → give the absolute path to the vendored `SKILL.md` and let the worker
  read it. The knowledge is a file in both cases; the Skill tool is the delivery mechanism
  that depends on how each deployment happened to install things.
- A failure to READ is worth logging; it should not abort generation. A failure to invoke a
  skill that was never installed is not evidence about the operator.

**Do not** conclude from "the pointer reached the brief" that the pointed-at thing can be
acted on. Those are different claims. The producer that surfaced these routes was verified to
reach the brief and shipped as if the second had also been checked — it had not, and the
routes named a skill no config dir on the host could load.

**Cross-reference**: OL-321 (a criterion narrower than the conclusion drawn from it — same
family, one level down); DEBT-258 (COMPENSATED-BY-CRITIC: knowledge that reaches a worker
this run and nothing makes it reach the next one).
