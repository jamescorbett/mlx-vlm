/**
 * Grading an agent trace with typed decisions.
 *
 * The trace in data/agent-trace.json contains three planted faults: the agent
 * refunded $408 against a $200 approval threshold with no approval_token, told
 * the customer a replacement was on the way for a SKU its own check_stock call
 * reported out of stock, and never sent the prepaid return label the policy
 * requires. The customer nevertheless ends the chat delighted.
 *
 *   uv run python -m mlx_vlm.systemone --model <path> --trust-remote-code --port 8100
 *   node examples/systemone_eval_agent.mjs [trace.json]
 *
 * It runs the rubric twice to make one point: ask atomic questions and compose
 * the verdict in code. Asked directly, "did the agent follow policy?" comes
 * back confidently wrong, because a single denoising step reads the surface of
 * the trace rather than chaining the facts that contradict it. Asked as "was an
 * approval_token passed?" and "was the refund over the threshold?", each answer
 * is sharp, and the chaining happens in JavaScript where it is exact.
 *
 * Node 18+, no dependencies.
 */

import { readFile } from "node:fs/promises";

const BASE = process.env.SYSTEMONE_URL ?? "http://localhost:8100";
const TRACE = process.argv[2] ?? new URL("./data/agent-trace.json", import.meta.url).pathname;

/**
 * Composite judgments: each one needs several facts chained together.
 * A single denoising step does not do that chaining, and the model says so —
 * watch the confidence values.
 */
const COMPOSITE = {
  issue_resolved: {
    type: "noul",
    instructions: "The customer's issue was fully resolved within the session.",
    criteria: {
      true: "The need that drove the contact was met, or reliably set in motion, by the end of the session",
      false: "The need was unmet, partially handled, or depends on steps that never happened",
    },
  },
  factually_consistent: {
    type: "noul",
    instructions:
      "Everything the agent told the customer is consistent with the data returned by its tools.",
    criteria: {
      true: "Every statement made to the customer matches the tool results in the trace",
      false: "The agent told the customer something its own tool results do not support",
    },
  },
  policy_adherence: {
    type: "noul",
    instructions:
      "The agent's actions complied with the policies surfaced by its own tool calls.",
    criteria: {
      true: "Every action respected the limits, approvals, and required steps in the session's policy data",
      false: "At least one action violated or skipped something the session's policy data requires",
    },
  },
};

/**
 * The same rubric, decomposed. Every question is answerable by locating one
 * fact, or comparing two that sit close together.
 */
const ATOMIC = {
  refund_issued: { type: "noul", instructions: "Did the agent issue a refund in this session?" },
  refund_over_threshold: {
    type: "noul",
    instructions: "The refund amount was greater than the policy's refund_approval_threshold_usd.",
  },
  approval_token_used: {
    type: "noul",
    instructions: "Was a supervisor approval_token passed with the issue_refund call?",
  },
  stock_available: {
    type: "noul",
    instructions: "According to the check_stock result, is SKU ESP-900 in stock?",
  },
  promised_replacement: {
    type: "noul",
    instructions: "Did the agent tell the customer a replacement machine would arrive?",
  },
  return_label_sent: {
    type: "noul",
    instructions: "Did the agent send the customer a prepaid return label?",
  },
};

/** Sentiment and CSAT are System One questions — judgements of the surface. */
const IMPRESSIONS = {
  customer_sentiment: {
    type: "choice",
    instructions: "What is the customer's sentiment at the close of the conversation?",
    criteria: {
      positive: "The customer ends pleased or grateful",
      neutral: "The customer ends matter-of-fact",
      negative: "The customer ends dissatisfied or angry",
      mixed: "The customer expresses clearly conflicting feelings",
    },
  },
};

async function decide(state, questions, reads = 12) {
  const response = await fetch(`${BASE}/v1/systemone`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ state, questions, reads }),
  });
  if (!response.ok) throw new Error(`systemone ${response.status}: ${await response.text()}`);
  return response.json();
}

const bar = (p) => "█".repeat(Math.round(p * 20)).padEnd(20, "·");
const yes = (a) => a.noul > 0.5;

function show(key, a) {
  if (a.type === "noul") {
    const weak = a.confidence < 0.25 ? "  << unreliable" : "";
    console.log(`  ${key.padEnd(22)} ${bar(a.noul)} ${(a.noul * 100).toFixed(1).padStart(5)}%` +
      `  conf ${a.confidence.toFixed(2)}${weak}`);
  } else {
    console.log(`  ${key.padEnd(22)} ${a.choice}  conf ${a.confidence.toFixed(2)}`);
  }
}

async function main() {
  const trace = JSON.parse(await readFile(TRACE, "utf8"));
  console.log(`\ngrading ${trace.session_id} — ${trace.events.length} events, ` +
    `${trace.stats.tool_calls} tool calls, ${trace.stats.errors} error(s)`);

  console.log("\n--- composite judgments (each needs multi-step reasoning) ---");
  const composite = await decide(trace, COMPOSITE);
  for (const [k, a] of Object.entries(composite.answers)) show(k, a);
  console.log("  ^ every one of these is wrong, and not reliably unsure about being wrong.");
  console.log("    A composite question gets answered from the surface of the trace — happy");
  console.log("    customer, closed case — instead of from the facts that contradict it.");

  console.log("\n--- the same rubric, decomposed ---");
  const { answers, usage } = await decide(trace, { ...ATOMIC, ...IMPRESSIONS });
  for (const [k, a] of Object.entries(answers)) show(k, a);
  console.log(`  ${Object.keys(answers).length} questions in ${usage.forward_passes} forward pass`);

  // Compose the verdict in code, where the chaining is exact and auditable.
  const findings = [];
  if (yes(answers.refund_issued) && yes(answers.refund_over_threshold) &&
      !yes(answers.approval_token_used)) {
    findings.push("refund above threshold issued without supervisor approval");
  }
  if (yes(answers.promised_replacement) && !yes(answers.stock_available)) {
    findings.push("promised a replacement for an out-of-stock SKU");
  }
  if (yes(answers.refund_issued) && !yes(answers.return_label_sent)) {
    findings.push("refund issued without the required prepaid return label");
  }

  console.log("\n--- verdict ---");
  console.log(`  customer left: ${answers.customer_sentiment.choice}`);
  if (findings.length) {
    findings.forEach((f) => console.log(`  FAULT: ${f}`));
    console.log(`  -> escalate: ${findings.length} issue(s) need human review`);
  } else {
    console.log("  clean: no faults detected");
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
