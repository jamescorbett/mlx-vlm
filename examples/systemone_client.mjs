/**
 * System One client — typed decisions from a diffusion model, no output parsing.
 *
 * Start the server first:
 *   uv run python -m mlx_vlm.systemone --model <path> --trust-remote-code --port 8100
 *
 * Then:
 *   node examples/systemone_client.mjs                 # text examples
 *   node examples/systemone_client.mjs path/to.png     # text + image examples
 *
 * Node 18+ (uses built-in fetch). No dependencies.
 */

import { readFile } from "node:fs/promises";
import { extname } from "node:path";

const BASE = process.env.SYSTEMONE_URL ?? "http://localhost:8100";

/** Post one state plus any number of typed questions; all answered in one pass. */
async function decide({ state = "", images, questions, reads = 4 }) {
  const response = await fetch(`${BASE}/v1/systemone`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ state, images, questions, reads }),
  });

  if (!response.ok) {
    // The server rejects questions it cannot reduce to one canvas slot, which
    // is worth surfacing rather than retrying — the question needs rewording.
    const detail = await response.text();
    throw new Error(`systemone ${response.status}: ${detail}`);
  }
  return response.json();
}

/** Browsers hand you data URLs from a file input; Node has to build one. */
async function toDataUrl(path) {
  const mime = { ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".gif": "image/gif", ".webp": "image/webp" }[extname(path).toLowerCase()];
  if (!mime) throw new Error(`Unsupported image type: ${path}`);
  return `data:${mime};base64,${(await readFile(path)).toString("base64")}`;
}

// Three helpers, one per primitive, that unwrap the answer you actually want.
const noul = (answers, key) => answers[key].noul;
const choice = (answers, key) => answers[key].choice;
const score = (answers, key) => {
  const a = answers[key];
  // A split distribution makes the mean meaningless; fall back to the modal level.
  return a.bimodal ? a.mode : a.score;
};

const pct = (p) => `${(p * 100).toFixed(1)}%`;

async function textExample() {
  console.log("\n=== text: one ticket, five questions, one forward pass ===");

  const { answers, usage } = await decide({
    reads: 8,
    state: `Hi, I have been trying to connect my Stripe account for 3 days and
nothing works. I have emailed support twice with no reply. This is blocking
our launch and I am seriously considering cancelling our subscription.`,
    questions: {
      urgency: { type: "noul", instructions: "Does this message express urgency?" },
      churnRisk: { type: "noul", instructions: "Is this customer at risk of cancelling?" },
      team: {
        type: "choice",
        instructions: "Which team should handle this?",
        criteria: {
          billing: "payments, invoices, refunds",
          integrations: "third-party connection problems",
          sales: "new purchase enquiries",
        },
      },
      severity: {
        type: "score",
        instructions: "How severe is this issue?",
        criteria: ["low", "medium", "high"],
      },
    },
  });

  console.log(`  urgency      ${pct(noul(answers, "urgency"))}`);
  console.log(`  churn risk   ${pct(noul(answers, "churnRisk"))}`);
  console.log(`  route to     ${choice(answers, "team")}`);
  console.log(`  severity     ${score(answers, "severity").toFixed(2)} / 2`);
  console.log(`  -> ${Object.keys(answers).length} questions, ${usage.forward_passes} forward pass`);

  // Branch on a decision the way you would on any other typed value.
  if (noul(answers, "churnRisk") > 0.8 && noul(answers, "urgency") > 0.8) {
    console.log("  ACTION: escalate to a human now");
  }
}

async function structuredStateExample() {
  console.log("\n=== structured state: `state` takes an object, not just a string ===");

  const { answers } = await decide({
    reads: 8,
    state: {
      pr: 481,
      title: "refactor auth middleware",
      files_changed: 34,
      additions: 1290,
      deletions: 12,
      tests_added: 0,
      ci: "failing",
    },
    questions: {
      needsReview: { type: "noul", instructions: "Does this PR need careful human review?" },
      hasTests: { type: "noul", instructions: "Does this PR include tests?" },
    },
  });

  console.log(`  needs review ${pct(noul(answers, "needsReview"))}`);
  console.log(`  has tests    ${pct(noul(answers, "hasTests"))}`);
}

async function cacheExample() {
  console.log("\n=== reusing a state: the second call skips the prefill ===");

  const state = `Refund policy. Refunds are allowed within 30 days of purchase.
Digital goods are non-refundable once downloaded. Shipping is never refunded.`;

  for (const [label, instructions] of [
    ["first ", "Are digital goods refundable after download?"],
    ["second", "Is shipping ever refunded?"],
  ]) {
    const started = Date.now();
    const { answers, usage } = await decide({
      state,
      reads: 4,
      questions: { q: { type: "noul", instructions } },
    });
    console.log(
      `  ${label}  ${pct(noul(answers, "q"))}  ${Date.now() - started}ms` +
        `  cached_input_tokens=${usage.cached_input_tokens}`,
    );
  }
}

async function imageExample(path) {
  console.log(`\n=== image: ${path} ===`);

  const { answers, usage } = await decide({
    reads: 8,
    images: [await toDataUrl(path)],
    questions: {
      red: { type: "noul", instructions: "Is this image mostly red?" },
      green: { type: "noul", instructions: "Is this image mostly green?" },
      shape: {
        type: "choice",
        instructions: "What shape is in this image?",
        criteria: { circle: null, square: null, triangle: null, nothing: "no clear shape" },
      },
    },
  });

  console.log(`  mostly red   ${pct(noul(answers, "red"))}`);
  console.log(`  mostly green ${pct(noul(answers, "green"))}`);
  console.log(`  shape        ${choice(answers, "shape")}`);
  console.log(`  -> ${usage.forward_passes} forward pass over ${usage.input_tokens} tokens`);
}

async function main() {
  try {
    const health = await (await fetch(`${BASE}/health`)).json();
    console.log(`connected to ${BASE} (${health.model})`);
  } catch {
    console.error(`No server at ${BASE}. Start it with:\n` +
      `  uv run python -m mlx_vlm.systemone --model <path> --trust-remote-code --port 8100`);
    process.exit(1);
  }

  await textExample();
  await structuredStateExample();
  await cacheExample();

  const image = process.argv[2];
  if (image) {
    await imageExample(image);
  } else {
    console.log("\n(pass an image path to run the image example)");
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
